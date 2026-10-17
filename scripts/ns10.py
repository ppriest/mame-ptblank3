"""ptblank3 type-2 decrypter model, mirrors ns10crypt.cpp / namcos10.cpp."""
import numpy as np

EMASK = [0x0000000057001200,0x00000000000000a4,0x0000000000000150,0x0000000000008004,
         0x0000000057002204,0x0000000009001000,0x0000000000102000,0x0000000064004888,
         0x0000000000100008,0x000000a0c0980600,0x000000001801c020,0x00005810881a09c0,
         0x0000000400980600,0x0000000000000300,0x0000000000000080,0x0000000000a00c0a]
DMASK = [0x0000000041051000,0x0000000000000024,0x0000000000000050,0x0000000000008004,
         0x0000000041052200,0x0000000009401100,0x0000000000122000,0x0000000004100898,
         0x0000000000120088,0x0000000040a80400,0x0000000019004000,0x000001859830281d,
         0x0000002400880201,0x0000000000000300,0x00000000000000c0,0x0000000000000408]
XMASK = 0xe1b8
INIT_SBOX = [0, 12, 13, 6, 2, 4, 9, 8, 11, 1, 7, 15, 10, 5, 14, 3]
NL_BIT = 11

def parity(x):
    return bin(x).count('1') & 1

def iv_word(iv):
    # bitswap(INIT_SBOX[iv], 3,16,16,2,1,16,16,0,16*8): source bit 3->out15, 2->out12, 1->out11, 0->out8
    s = INIT_SBOX[iv]
    return (((s >> 3) & 1) << 15) | (((s >> 2) & 1) << 12) | (((s >> 1) & 1) << 11) | ((s & 1) << 8)

class Decrypter:
    def init(self, iv):
        self.pc = iv_word(iv)   # previous cipherwords (64-bit shift register)
        self.pp = 0
        self.mask = 0
    def decrypt(self, c, nl_bit):
        p = c ^ self.mask
        self.pc = ((self.pc << 16) ^ c) & 0xffffffffffffffff
        self.pp = ((self.pp << 16) ^ p) & 0xffffffffffffffff
        m = 0
        for j in range(15, -1, -1):
            m = (m << 1) ^ parity(EMASK[j] & self.pc) ^ parity(DMASK[j] & self.pp)
        m ^= XMASK
        m ^= nl_bit << NL_BIT
        self.mask = m
        return p

def load_dump(path):
    raw = np.fromfile(path, dtype=np.uint8).reshape(-1, 10)
    typ = raw[:, 0].view('S1')
    b = raw[:, 1].astype(np.uint8)
    c = (raw[:, 2].astype(np.uint16) | (raw[:, 3].astype(np.uint16) << 8))
    p = (raw[:, 4].astype(np.uint16) | (raw[:, 5].astype(np.uint16) << 8))
    cnt = raw[:, 6:10].copy().view(np.uint32).ravel()
    return typ, b, c, p, cnt
