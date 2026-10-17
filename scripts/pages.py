import re, numpy as np
def page_sequence(logpath):
    """NAND page address (m_nand_address >> 8, 16 bits: row | page<<8) for each page read while the decrypter is active."""
    pages = []; active = False
    for line in open(logpath, encoding='utf-8', errors='replace'):
        if 'crypto_switch_w: 8000' in line: active = True; continue
        if 'crypto_switch_w: 0000' in line: break
        if active:
            m = re.search(r'nand_address_page_w [0-9a-f]{2} ([0-9a-f]{8})', line)
            if m: pages.append(int(m.group(1), 16) >> 8)
    return np.array(pages, np.int64)
