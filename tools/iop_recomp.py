#!/usr/bin/env python3
"""Recompiles an IOP module (IRX, MIPS R3000A) to native C++, from the image the IOP loaded (PS2X_IOP_DUMP_MODULES:
<name>_<base>.bin relocated at its base, with <name>_<base>.json: base, size, entry, gp, file hash).

  tools/iop_recomp.py <module>.json [--out <dir>]    writes <module>.cpp (and prints what it found)
  tools/iop_recomp.py <module>.json --list           functions, exports, imports, what is left to the interpreter
  tools/iop_recomp.py <module>.json --entry 0x...,0x...   more function starts (only reached through pointers)

Functions come from the module's entry, its export tables (magic 0x41C00000), the targets of JALs, the addresses
the code builds with LUI + ADDIU/ORI that land on a function prologue (ADDIU $sp, $sp, -n: SIF command handlers,
thread entries and callbacks a module only passes as pointers), the data words that hold such an address, every
prologue right after a function's end, the --entry list, and the functions those reach. Import stubs (magic 0x41E00000: `jr ra; addiu $0, $0, ordinal`) stay calls into the IOP's
import dispatch. A function the tool cannot translate safely (an unknown instruction, a load whose register the next
instruction reads: a load delay slot the R3000 would honour) is left to the interpreter.
"""
import json
import os
import struct
import sys

EXPORT_MAGIC = 0x41C00000
IMPORT_MAGIC = 0x41E00000


class Module:
    def __init__(self, json_path):
        with open(json_path) as f:
            info = json.load(f)
        self.name = info['name']
        self.base = info['base']
        self.size = info['size']
        self.entry = info['entry']
        self.gp = info['gp']
        self.hash = info['hash']
        with open(os.path.splitext(json_path)[0] + '.bin', 'rb') as f:
            self.image = f.read()

    def word(self, addr):
        off = addr - self.base
        if off < 0 or off + 4 > len(self.image):
            return None
        return struct.unpack_from('<I', self.image, off)[0]

    def contains(self, addr):
        return self.base <= addr < self.base + self.size

    def tables(self):
        """(exports {addr: (library, index)}, import stubs {addr: (library, ordinal)})."""
        exports, imports = {}, {}
        for off in range(0, len(self.image) - 20, 4):
            magic = struct.unpack_from('<I', self.image, off)[0]
            if magic not in (EXPORT_MAGIC, IMPORT_MAGIC):
                continue
            library = self.image[off + 12:off + 20].split(b'\0')[0].decode('latin-1')
            if not library.isprintable() or not library:
                continue
            addr = self.base + off + 20
            if magic == EXPORT_MAGIC:
                index = 0
                while True:
                    target = self.word(addr + 4 * index)
                    if not target or not self.contains(target):
                        break
                    exports.setdefault(target, (library, index))
                    index += 1
            else:
                while True:
                    jr, addiu = self.word(addr), self.word(addr + 4)
                    if jr != 0x03E00008 or addiu is None or (addiu >> 16) != 0x2400:
                        break
                    imports[addr] = (library, addiu & 0xffff)
                    addr += 8
        return exports, imports


# ---------------------------------------------------------------------------------------------------------------
# Decoding

LOADS = {0x20: ('LB', 1), 0x21: ('LH', 2), 0x22: ('LWL', 4), 0x23: ('LW', 4), 0x24: ('LBU', 1), 0x25: ('LHU', 2),
         0x26: ('LWR', 4)}
STORES = {0x28: ('SB', 1), 0x29: ('SH', 2), 0x2A: ('SWL', 4), 0x2B: ('SW', 4), 0x2E: ('SWR', 4)}


class Insn:
    __slots__ = ('addr', 'word', 'op', 'rs', 'rt', 'rd', 'sa', 'funct', 'imm', 'simm', 'target', 'kind', 'reads',
                 'writes', 'load_reg')

    def __init__(self, addr, word):
        self.addr, self.word = addr, word
        self.op, self.rs, self.rt = word >> 26, (word >> 21) & 31, (word >> 16) & 31
        self.rd, self.sa, self.funct = (word >> 11) & 31, (word >> 6) & 31, word & 63
        self.imm = word & 0xffff
        self.simm = self.imm - 0x10000 if self.imm & 0x8000 else self.imm
        self.target = ((addr + 4) & 0xF0000000) | ((word & 0x03FFFFFF) << 2)
        self.kind, self.reads, self.writes, self.load_reg = decode(self)


def decode(i):
    """(kind, registers read, registers written, register loaded with a delay). kind: 'alu', 'load', 'store',
    'branch', 'j', 'jal', 'jr', 'jalr', 'exit' (left to the interpreter: native code stops there)."""
    op, rs, rt, rd, f = i.op, i.rs, i.rt, i.rd, i.funct
    if i.word == 0:
        return 'alu', (), (), None
    if op == 0:
        if f in (0x00, 0x02, 0x03):
            return 'alu', (rt,), (rd,), None
        if f in (0x04, 0x06, 0x07, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x2A, 0x2B):
            return 'alu', (rs, rt), (rd,), None
        if f == 0x08:
            return 'jr', (rs,), (), None
        if f == 0x09:
            return 'jalr', (rs,), (rd,), None
        if f in (0x10, 0x12):
            return 'alu', (), (rd,), None  # MFHI / MFLO (hi, lo are not tracked as hazards)
        if f in (0x11, 0x13):
            return 'alu', (rs,), (), None  # MTHI / MTLO
        if f in (0x18, 0x19, 0x1A, 0x1B):
            return 'alu', (rs, rt), (), None
        return 'exit', (), (), None  # SYSCALL, BREAK (gcc's divide-by-zero trap), unknown: the interpreter's
    if op == 1:
        if rt in (0, 1):
            return 'branch', (rs,), (), None
        if rt in (16, 17):
            return 'branch', (rs,), (31,), None  # BLTZAL / BGEZAL
        return 'exit', (), (), None
    if op == 2:
        return 'j', (), (), None
    if op == 3:
        return 'jal', (), (31,), None
    if op in (4, 5):
        return 'branch', (rs, rt), (), None
    if op in (6, 7):
        return 'branch', (rs,), (), None
    if op in (8, 9, 0xA, 0xB, 0xC, 0xD, 0xE):
        return 'alu', (rs,), (rt,), None
    if op == 0xF:
        return 'alu', (), (rt,), None
    if op == 0x10 and rs == 0:
        return 'load', (), (), rt  # MFC0: a delayed load too
    if op in LOADS:
        name = LOADS[op][0]
        return 'load', (rs, rt) if name in ('LWL', 'LWR') else (rs,), (), rt
    if op in STORES:
        return 'store', (rs, rt), (), None
    return 'exit', (), (), None  # MTC0, RFE, COP2, unknown: the interpreter runs it


# ---------------------------------------------------------------------------------------------------------------
# Functions

class Function:
    def __init__(self, start):
        self.start = start
        self.insns = {}       # addr -> Insn (reachable ones)
        self.labels = set()   # branch targets inside the function
        self.resume = {start}  # where the runtime may enter it (entry, after calls, loop heads)
        self.calls = set()    # JAL targets
        self.unsafe = None    # why it stays interpreted


def explore(module, start, limit):
    """The instructions reachable from start without leaving [start, limit) (branches, delay slots)."""
    fn = Function(start)
    work = [start]
    while work:
        addr = work.pop()
        while start <= addr < limit and addr not in fn.insns:
            word = module.word(addr)
            if word is None:
                fn.unsafe = f'runs off the image at {addr:#x}'
                return fn
            insn = Insn(addr, word)
            fn.insns[addr] = insn
            if insn.kind == 'exit':
                break  # native code stops there and the interpreter runs it (and what follows)
            if insn.kind in ('branch', 'j', 'jal', 'jr', 'jalr'):
                slot = module.word(addr + 4)
                if slot is None:
                    fn.unsafe = f'delay slot off the image at {addr:#x}'
                    return fn
                slot_insn = Insn(addr + 4, slot)
                if slot_insn.kind in ('branch', 'j', 'jal', 'jr', 'jalr', 'exit'):
                    fn.unsafe = f'jump or unsafe instruction in a delay slot at {addr + 4:#x}'
                    return fn
                fn.insns[addr + 4] = slot_insn
                if insn.kind == 'branch':
                    target = addr + 4 + (insn.simm << 2)
                    if start <= target < limit:
                        fn.labels.add(target)
                        if target <= addr:
                            fn.resume.add(target)
                        work.append(target)
                    addr += 8
                    continue
                if insn.kind == 'j':
                    if start <= insn.target < limit:
                        fn.labels.add(insn.target)
                        if insn.target <= addr:
                            fn.resume.add(insn.target)
                        work.append(insn.target)
                    break
                if insn.kind in ('jal', 'jalr'):
                    if insn.kind == 'jal':
                        fn.calls.add(insn.target)
                    fn.resume.add(addr + 8)
                    fn.labels.add(addr + 8)
                    addr += 8
                    continue
                break  # jr
            addr += 4
    # A load whose register the next instruction reads: the R3000 would give it the old value (its load delay).
    for addr, insn in fn.insns.items():
        if insn.load_reg:
            nxt = fn.insns.get(addr + 4)
            if nxt and insn.load_reg in nxt.reads and not (insn.op in (0x22, 0x26) and nxt.op in (0x22, 0x26)):
                fn.unsafe = f'load delay hazard at {addr:#x}'
                return fn
    return fn


def pointer_seeds(module):
    """Function starts the code only takes the address of: LUI rt, hi then, within the next few instructions,
    ADDIU/ORI rt2, rt, lo giving an address in the module whose first instruction is ADDIU $sp, $sp, -n. Also the
    data words that hold such an address (function tables, structures of callbacks), and every prologue right after
    a function's end (JR $ra, its delay slot, nop padding): the functions only other unreached code or a jump table
    leads to (Jade Cocoon 2's M2SOUND.IRX: 21 functions from the LUI pairs alone, 89 with these)."""
    def is_prologue(target):
        prologue = module.word(target) if module.contains(target) and target % 4 == 0 else None
        return prologue is not None and prologue >> 16 == 0x27BD and prologue & 0x8000

    seeds = set()
    for addr in range(module.base, module.base + module.size, 4):
        insn = module.word(addr)
        if insn is None:
            continue
        if is_prologue(insn):
            seeds.add(insn)
        if is_prologue(addr):
            back = addr - 8
            while module.contains(back) and module.word(back) == 0 and module.word(back + 4) == 0:
                back -= 4
            if module.word(back) == 0x03E00008:
                seeds.add(addr)
        if insn >> 26 != 0x0F:
            continue
        rt, hi = (insn >> 16) & 31, insn & 0xFFFF
        for follow in range(addr + 4, addr + 36, 4):
            low_insn = module.word(follow)
            if low_insn is None:
                break
            op, rs = low_insn >> 26, (low_insn >> 21) & 31
            if op in (0x09, 0x0D) and rs == rt:
                lo = low_insn & 0xFFFF
                target = ((hi << 16) + (lo - 0x10000 if op == 0x09 and lo & 0x8000 else lo)) & 0xFFFFFFFF
                prologue = module.word(target) if module.contains(target) and target % 4 == 0 else None
                if prologue is not None and prologue >> 16 == 0x27BD and prologue & 0x8000:
                    seeds.add(target)
                break
            if (low_insn >> 16) & 31 == rt and op not in (0x2B, 0x29, 0x28):  # rt written again (not a store)
                break
    return seeds


def find_functions(module, exports, imports, extra=()):
    code_end = module.base + module.size
    seeds = {module.entry} | set(exports) | pointer_seeds(module) | set(extra)
    functions = {}
    while True:
        bounds = sorted(seeds)
        new = set()
        for index, start in enumerate(bounds):
            if start in functions or start in imports or not module.contains(start):
                continue
            limit = bounds[index + 1] if index + 1 < len(bounds) else code_end
            fn = explore(module, start, limit)
            functions[start] = fn
            for target in fn.calls:
                if module.contains(target) and target not in imports and target not in seeds:
                    new.add(target)
        if not new:
            break
        seeds |= new
        # A new seed inside an explored function cuts it: explore those again with their new limit.
        bounds = sorted(seeds)
        for index, start in enumerate(bounds):
            limit = bounds[index + 1] if index + 1 < len(bounds) else code_end
            fn = functions.get(start)
            if fn and any(addr >= limit for addr in fn.insns):
                del functions[start]
    return functions


# ---------------------------------------------------------------------------------------------------------------
# C++

def r(n):
    return '0u' if n == 0 else f'c.gpr[{n}]'


def w(n, value):
    return '' if n == 0 else f'c.gpr[{n}] = {value};'


def alu(i):
    op, rs, rt, rd, sa, f, imm, simm = i.op, i.rs, i.rt, i.rd, i.sa, i.funct, i.imm, i.simm
    if i.word == 0:
        return ''
    if op == 0:
        S = {0x00: f'{r(rt)} << {sa}', 0x02: f'{r(rt)} >> {sa}', 0x03: f'(uint32_t)((int32_t){r(rt)} >> {sa})',
             0x04: f'{r(rt)} << ({r(rs)} & 31u)', 0x06: f'{r(rt)} >> ({r(rs)} & 31u)',
             0x07: f'(uint32_t)((int32_t){r(rt)} >> ({r(rs)} & 31u))',
             0x20: f'{r(rs)} + {r(rt)}', 0x21: f'{r(rs)} + {r(rt)}', 0x22: f'{r(rs)} - {r(rt)}',
             0x23: f'{r(rs)} - {r(rt)}', 0x24: f'{r(rs)} & {r(rt)}', 0x25: f'{r(rs)} | {r(rt)}',
             0x26: f'{r(rs)} ^ {r(rt)}', 0x27: f'~({r(rs)} | {r(rt)})',
             0x2A: f'(uint32_t)((int32_t){r(rs)} < (int32_t){r(rt)})', 0x2B: f'(uint32_t)({r(rs)} < {r(rt)})'}
        if f in S:
            return w(rd, S[f])
        if f == 0x10:
            return w(rd, 'c.hi')
        if f == 0x12:
            return w(rd, 'c.lo')
        if f == 0x11:
            return f'c.hi = {r(rs)};'
        if f == 0x13:
            return f'c.lo = {r(rs)};'
        if f == 0x18:
            return (f'{{ const int64_t p = (int64_t)(int32_t){r(rs)} * (int32_t){r(rt)}; c.lo = (uint32_t)p; '
                    f'c.hi = (uint32_t)((uint64_t)p >> 32); }}')
        if f == 0x19:
            return (f'{{ const uint64_t p = (uint64_t){r(rs)} * {r(rt)}; c.lo = (uint32_t)p; '
                    f'c.hi = (uint32_t)(p >> 32); }}')
        if f == 0x1A:
            return f'divide(c, {r(rs)}, {r(rt)});'
        if f == 0x1B:
            return f'divideUnsigned(c, {r(rs)}, {r(rt)});'
    I = {8: f'{r(rs)} + {simm & 0xffffffff:#x}u', 9: f'{r(rs)} + {simm & 0xffffffff:#x}u',
         0xA: f'(uint32_t)((int32_t){r(rs)} < {simm})', 0xB: f'(uint32_t)({r(rs)} < {simm & 0xffffffff:#x}u)',
         0xC: f'{r(rs)} & {imm:#x}u', 0xD: f'{r(rs)} | {imm:#x}u', 0xE: f'{r(rs)} ^ {imm:#x}u',
         0xF: f'{imm << 16:#x}u'}
    return w(rt, I[op])


def memory(i):
    op, rs, rt = i.op, i.rs, i.rt
    addr = f'{r(rs)} + {i.simm & 0xffffffff:#x}u'
    if op == 0x10:
        return w(rt, f'c.cop0[{i.rd}]')
    if op in LOADS:
        name = LOADS[op][0]
        value = {'LB': f'(uint32_t)(int32_t)(int8_t)e.r8({addr})', 'LBU': f'e.r8({addr})',
                 'LH': f'(uint32_t)(int32_t)(int16_t)e.r16({addr})', 'LHU': f'e.r16({addr})',
                 'LW': f'e.r32({addr})', 'LWL': f'loadLeft(e, {addr}, {r(rt)})',
                 'LWR': f'loadRight(e, {addr}, {r(rt)})'}[name]
        return w(rt, value)
    name = STORES[op][0]
    return {'SB': f'e.w8({addr}, (uint8_t){r(rt)});', 'SH': f'e.w16({addr}, (uint16_t){r(rt)});',
            'SW': f'e.w32({addr}, {r(rt)});', 'SWL': f'storeLeft(e, {addr}, {r(rt)});',
            'SWR': f'storeRight(e, {addr}, {r(rt)});'}[name]


def simple(i):
    return memory(i) if i.kind in ('load', 'store') else alu(i)


def branch_condition(i):
    rs, rt = r(i.rs), r(i.rt)
    if i.op == 1:
        return {0: f'(int32_t){rs} < 0', 1: f'(int32_t){rs} >= 0', 16: f'(int32_t){rs} < 0',
                17: f'(int32_t){rs} >= 0'}[i.rt]
    return {4: f'{rs} == {rt}', 5: f'{rs} != {rt}', 6: f'(int32_t){rs} <= 0', 7: f'(int32_t){rs} > 0'}[i.op]


def name_of(module, start):
    return f'iop_{module.hash[:8]}_{start:06x}'


def emit_function(module, fn, native):
    out = [f'void {name_of(module, fn.start)}(Env &e)', '{', '    NativeCpu &c = e.cpu;', '    switch (c.pc)', '    {']
    for addr in sorted(fn.resume):
        out.append(f'    case {addr:#x}u: goto L_{addr:x};')
    out += ['    default: return; // not an entry of this function: the interpreter goes on', '    }']
    order = sorted(fn.insns)
    skip = set()
    for addr in order:
        if addr in skip:
            continue
        i = fn.insns[addr]
        if addr in fn.labels or addr in fn.resume:
            out.append(f'L_{addr:x}:')
        if i.kind == 'exit':
            out.append(f'    c.pc = {addr:#x}u; return; // {i.word:#010x}: the interpreter runs it')
            continue
        # Each instruction counts as one, as in the interpreter.
        if i.kind in ('alu', 'load', 'store'):
            out.append(f'    e.tick(1); {simple(i)}'.rstrip())
            continue
        slot = fn.insns[addr + 4]
        skip.add(addr + 4)
        slot_code = simple(slot)
        if i.kind == 'branch':
            target = addr + 4 + (i.simm << 2)
            link = w(31, f'{addr + 8:#x}u') if i.rt in (16, 17) and i.op == 1 else ''
            out.append(f'    {{ e.tick(2); const bool taken = {branch_condition(i)}; {link} {slot_code}')
            if target in fn.insns:
                backward = target <= addr
                if backward:
                    out.append(f'      if (taken) {{ if (e.expired()) {{ c.pc = {target:#x}u; return; }} goto L_{target:x}; }} }}')
                else:
                    out.append(f'      if (taken) goto L_{target:x}; }}')
            else:
                out.append(f'      if (taken) {{ c.pc = {target:#x}u; return; }} }}')
            continue
        if i.kind == 'j':
            out.append(f'    e.tick(2); {slot_code}')
            if i.target in fn.insns:
                if i.target <= addr:
                    out.append(f'    if (e.expired()) {{ c.pc = {i.target:#x}u; return; }}')
                out.append(f'    goto L_{i.target:x};')
            else:
                out.append(f'    c.pc = {i.target:#x}u; return; // a jump out: the caller goes on there')
            continue
        if i.kind == 'jal':
            out.append(f'    e.tick(2); c.gpr[31] = {addr + 8:#x}u; {slot_code}')
            out.append(f'    if (!e.call({i.target:#x}u, {addr + 8:#x}u)) return;')
            continue
        if i.kind == 'jalr':
            link = w(i.rd, f'{addr + 8:#x}u')
            out.append(f'    {{ e.tick(2); const uint32_t target = {r(i.rs)}; {link} {slot_code}')
            out.append(f'      if (!e.call(target, {addr + 8:#x}u)) return; }}')
            continue
        if i.kind == 'jr':
            out.append(f'    {{ e.tick(2); const uint32_t target = {r(i.rs)}; {slot_code}')
            out.append('      c.pc = target; return; }')
            continue
    out += ['    c.pc = %#xu; // past the end' % (order[-1] + 4), '}', '']
    return out


def emit_module(module, functions):
    native = {start: fn for start, fn in functions.items() if not fn.unsafe}
    lines = ['// Generated by tools/iop_recomp.py from %s (base %#x, size %#x, file hash %s): do not edit.'
             % (module.name, module.base, module.size, module.hash),
             '#include "ps2x/iop/iop_native.h"', '', 'namespace', '{',
             '    using namespace ps2x::iop::native;', '']
    for start in sorted(native):
        lines.append(f'void {name_of(module, start)}(Env &e);')
    lines.append('')
    for start in sorted(native):
        lines += emit_function(module, native[start], native)
    entries = []
    for start in sorted(native):
        for addr in sorted(native[start].resume):
            entries.append(f'    {{{addr:#x}u, &{name_of(module, start)}}},')
    lines += ['const Entry kEntries[] = {'] + entries + ['};', '',
              f'const Registration kModule("{module.name}", 0x{module.hash}ull, {module.base:#x}u, {module.size:#x}u, '
              f'kEntries, sizeof(kEntries) / sizeof(kEntries[0]));', '}']
    return '\n'.join(lines) + '\n'


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    module = Module(args[0])
    exports, imports = module.tables()
    print(f'{module.name}: base {module.base:#x} size {module.size:#x} entry {module.entry:#x} gp {module.gp:#x}')
    print(f'  {len(exports)} exported functions, {len(imports)} import stubs '
          f'({len(set(lib for lib, _ in imports.values()))} libraries)')
    extra = []
    if '--entry' in args:
        extra = [int(value, 16) for value in args[args.index('--entry') + 1].split(',') if value]
    functions = find_functions(module, exports, imports, extra)
    native = [fn for fn in functions.values() if not fn.unsafe]
    print(f'  {len(functions)} functions, {len(native)} recompiled, {len(functions) - len(native)} left to the interpreter')
    if '--out' in args:
        out_dir = args[args.index('--out') + 1]
        path = os.path.join(out_dir, os.path.splitext(module.name)[0].lower() + '_iop.cpp')
        with open(path, 'w', newline='\n') as f:
            f.write(emit_module(module, functions))
        print(f'  wrote {path}')
    if '--list' in args:
        for start, fn in sorted(functions.items()):
            print(f'  function {start:#x}: {len(fn.insns)} instructions' + (f', interpreted ({fn.unsafe})' if fn.unsafe else ''))
        for addr, (lib, index) in sorted(exports.items()):
            print(f'  export {lib}[{index}] at {addr:#x}')
        for addr, (lib, ordinal) in sorted(imports.items()):
            print(f'  import {lib}:{ordinal} stub at {addr:#x}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
