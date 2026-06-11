"""Single-file container for FCGS bitstream directories.

Format (little-endian):
    b'FCGS' | uint8 version=1 | uint32 entry_count
    per entry: uint16 name_len | name utf-8 (path relative to packed dir) | uint64 offset | uint64 size
    concatenated file blobs

Usage:
    python tools/fcgs_container.py pack   <bitstream_dir> <out.fcgs>
    python tools/fcgs_container.py unpack <in.fcgs> <dst_dir>
"""
import os
import struct
import sys

MAGIC = b'FCGS'
VERSION = 1


def pack(src_dir, out_path):
    entries = []
    for root, _, files in os.walk(src_dir):
        for fn in sorted(files):
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, src_dir)
            entries.append((rel, os.path.getsize(full)))
    entries.sort()

    header = MAGIC + struct.pack('<BI', VERSION, len(entries))
    names = [rel.encode('utf-8') for rel, _ in entries]
    table_size = sum(2 + len(n) + 8 + 8 for n in names)
    offset = len(header) + table_size
    table = b''
    for (rel, size), name in zip(entries, names):
        table += struct.pack('<H', len(name)) + name + struct.pack('<QQ', offset, size)
        offset += size

    with open(out_path, 'wb') as fout:
        fout.write(header)
        fout.write(table)
        for rel, _ in entries:
            with open(os.path.join(src_dir, rel), 'rb') as fin:
                fout.write(fin.read())
    return len(entries), os.path.getsize(out_path)


def unpack(container_path, dst_dir):
    with open(container_path, 'rb') as fin:
        assert fin.read(4) == MAGIC, f'{container_path} is not an FCGS container'
        version, count = struct.unpack('<BI', fin.read(5))
        assert version == VERSION, f'unsupported container version {version}'
        entries = []
        for _ in range(count):
            name_len, = struct.unpack('<H', fin.read(2))
            rel = fin.read(name_len).decode('utf-8')
            offset, size = struct.unpack('<QQ', fin.read(16))
            entries.append((rel, offset, size))
        for rel, offset, size in entries:
            dst = os.path.join(dst_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            fin.seek(offset)
            with open(dst, 'wb') as fout:
                fout.write(fin.read(size))
    return len(entries)


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ('pack', 'unpack'):
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == 'pack':
        n, size = pack(sys.argv[2], sys.argv[3])
        print(f'packed {n} files into {sys.argv[3]} ({size} bytes)')
    else:
        n = unpack(sys.argv[2], sys.argv[3])
        print(f'unpacked {n} files into {sys.argv[3]}')


if __name__ == '__main__':
    main()
