---- environment ----

const
    bit32,
    buffer,
    vector,
    string
=
    bit32,
    buffer,
    vector,
    string;

const
    band,
    bor,
    bxor,
    bnot,
    lshift,
    rshift,
    countlz,
    countrz
=
    bit32.band,
    bit32.bor,
    bit32.bxor,
    bit32.bnot,
    bit32.lshift,
    bit32.rshift,
    bit32.countlz,
    bit32.countrz;

const format = string.format;
const char   = string.char;
const v      = vector.create;

---- definition ----

export type uint64 =            vector;
export type uint32 =            number;

type uint          = (uint64 | uint32);

---- interface ----

type bit64 = {
    --- constructors ---

    pair:
        & ((h: number, l: number)           ->         (uint64))
        & ((uint64)                         -> (number, number));

    buffer:
        & ((b: buffer, o: number?)          ->         (uint64))
        & ((uint64, b: buffer?, o: number?) ->        (buffer?));

    double:
        & ((n: number)                      ->         (uint64))
        & ((uint64)                         ->         (number));

    --- constants ---

    zero:                                                 uint64;

    --- arithmetic ---

    add: (
        n:                                                  uint,
        x:                                                  uint
    ) -> uint64;

    sub: (
        n:                                                  uint,
        x:                                                  uint
    ) -> uint64;

    mul: (
        n:                                                  uint,
        x:                                                  uint
    ) -> uint64;

    div: (
        n:                                                  uint,
        x:                                                  uint
    ) -> uint64;

    mod: (
        n:                                                  uint,
        x:                                                  uint
    ) -> uint64;

    pow: (
        n:                                                  uint,
        x:                                                  uint
    ) -> uint64;

    --- bitwise ---

    band: (
        ...                                                 uint
    ) -> uint64;

    bor: (
        ...                                                 uint
    ) -> uint64;

    bxor: (
        ...                                                 uint
    ) -> uint64;

    bnot: (
        n:                                                  uint
    ) -> uint64;

    lshift: (
        n:                                                  uint,
        d:                                                  uint
    ) -> uint64;

    rshift: (
        n:                                                  uint,
        d:                                                  uint
    ) -> uint64;

    lrotate: (
        n:                                                  uint,
        d:                                                  uint
    ) -> uint64;

    rrotate: (
        n:                                                  uint,
        d:                                                  uint
    ) -> uint64;

    countlz: (
        n:                                                  uint
    ) -> uint64;

    countrz: (
        n:                                                  uint
    ) -> uint64;

    --- comparison ---

    eq: (
        n:                                                  uint,
        x:                                                  uint
    ) -> boolean;

    lt: (
        n:                                                  uint,
        x:                                                  uint
    ) -> boolean;

    le: (
        n:                                                  uint,
        x:                                                  uint
    ) -> boolean;

    --- conversion ---

    hex: (
        n:                                                  uint
    ) -> string;

    decimal: (
        n:                                                  uint
    ) -> string;
};

---- constants ----

const
    m32, m22,
    m20, ll, lm
=
    4294967296, 0x3FFFFF,
    0x0FFFFF, 4194304, 1024;

---- functions ----

const function normalize(uint64: uint64): vector
    return v(
        band(uint64.x, m22),
        band(uint64.y, m20),
        band(uint64.z, m22)
    );
end;

---- module ----

const bit64 = {} :: bit64; do
    --- private ---

    const function uint64(n: uint): uint64
        return if('number' == type(n)) then bit64.double(n) else n :: uint64;
    end;

    const function shift(d: uint): number
        if('number' == type(d)) then return d end;

        local h, l = bit64.pair(d :: uint64);
        return if(h ~= 0) then 64 else l;
    end;

    --- constructors ---

    function bit64.pair(h, l)
        if('vector' == type(h)) then
            const
                x: number,
                y: number,
                z: number
            = h.x, h.y, h.z;

            return
                x * lm + y // lm,
                (y % lm) * ll + z;
        end;

        assert(
            'number' == type(h) and 'number' == type(l),
            'bit64.pair: expected (number, number) or uint64'
        );

        return v(
            h // lm,
            (h % lm) * lm + l // ll,
            l % ll
        );
    end;

    function bit64.buffer(b, o, n)
        if('vector' == type(b)) then
            const h, l  = bit64.pair(b);
            const block = if(not o) then buffer.create(8) else o;

            n = n or 0;

            buffer.writeu32(block,     n, h);
            buffer.writeu32(block, n + 4, l);

            return(block);
        end;

        if('number' == type(b)) then
            const h, l  = 0, b;
            const block = if(not o) then buffer.create(8) else o;

            n = n or 0;

            buffer.writeu32(block,     n, h);
            buffer.writeu32(block, n + 4, l);

            return(block);
        end;

        assert(
            'buffer' == type(b),
            'bit64.buffer: expected buffer or uint64'
        );

        o = o or 0;

        return bit64.pair(
            buffer.readu32(b :: buffer,       o :: number),
            buffer.readu32(b :: buffer, (o + 4) :: number)
        )
    end;

    function bit64.double(n)
        if('vector' == type(n)) then
            const h, l = bit64.pair(n);

            return h * m32 + l;
        end;

        assert(
            'number' == type(n),
            'bit64.double: expected (number) or uint64'
        );

        return bit64.pair(
            n // m32,
            n  % m32
        )
    end;

    --- constants ---

    bit64.zero = vector.zero;

    --- arithmetic ---

    function bit64.add(n, x)
        n = type(n) == 'number' and bit64.double(n) or n;
        x = type(x) == 'number' and bit64.double(x) or x;

        local prime = n + x;

        prime += v(
            0, (if prime.z > m22 then 1 else 0), 0
        );

        prime += v(
            (if prime.y > m20 then 1 else 0), 0, 0
        );

        return normalize(prime);
    end;

    function bit64.sub(n, x)
        n = type(n) == 'number' and bit64.double(n) or n;
        x = type(x) == 'number' and bit64.double(x) or x;

        local prime = n - x;

        prime -= v(
            0, (if prime.z < 0 then 1 else 0), 0
        );

        prime -= v(
            (if prime.y < 0 then 1 else 0), 0, 0
        );

        return normalize(prime);
    end;

    function bit64.mul(n, x)
        n = type(n) == 'number' and bit64.double(n) or n;
        x = type(x) == 'number' and bit64.double(x) or x;

        local prime = n * x;

        prime += v(
            0, (if prime.z > m22 then 1 else 0), 0
        );

        prime += v(
            (if prime.y > m20 then 1 else 0), 0, 0
        );

        return normalize(prime);
    end;

    function bit64.div(n, x)
        n = type(n) == 'number' and bit64.double(n) or n;
        x = type(x) == 'number' and bit64.double(x) or x;

        return normalize(n // x);
    end;

    function bit64.mod(n, x)
        n = type(n) == 'vector' and bit64.double(n) or n;
        x = type(x) == 'vector' and bit64.double(x) or x;

        return bit64.double(n % x);
    end;

    function bit64.pow(n, x)
        n = uint64(n);
        x = uint64(x);

        if(x.x == 0 and x.y == 0 and x.z == 0) then
            return bit64.pair(0, 1);
        end;

        if(x.x == 0 and x.y == 0 and x.z == 1) then
            return n;
        end;

        local result = bit64.pair(0, 1);

        while(true) do
            const _, xl = bit64.pair(x);

            if(band(xl, 1) ~= 0) then
                result = bit64.mul(result, n);
            end;

            x = bit64.rshift(x, 1);

            if(x.x == 0 and x.y == 0 and x.z == 0) then
                break;
            end;

            n = bit64.mul(n, n);
        end;

        return result;
    end;

    --- bitwise ---

    function bit64.band(...)
        const args = { ... };
        local r    = uint64(args[1]);

        for i = 2, #args do
            const x = uint64(args[i]);

            r = v(
                band(r.x, x.x),
                band(r.y, x.y),
                band(r.z, x.z)
            );
        end;

        return r;
    end;

    function bit64.bor(...)
        const args = { ... };
        local r    = uint64(args[1]);

        for i = 2, #args do
            const x = uint64(args[i]);

            r = v(
                bor(r.x, x.x),
                bor(r.y, x.y),
                bor(r.z, x.z)
            );
        end;

        return r;
    end;

    function bit64.bxor(...)
        const args = { ... };
        local r    = uint64(args[1]);

        for i = 2, #args do
            const x = uint64(args[i]);

            r = v(
                bxor(r.x, x.x),
                bxor(r.y, x.y),
                bxor(r.z, x.z)
            );
        end;

        return r;
    end;

    function bit64.bnot(n)
        n = uint64(n);

        return v(
            band(bnot(n.x), m22),
            band(bnot(n.y), m20),
            band(bnot(n.z), m22)
        );
    end;

    function bit64.lshift(n, d)
        n = uint64(n);

        const s = shift(d);

        if(s <= 0) then
            return normalize(n);
        end;

        if(s >= 64) then
            return bit64.zero;
        end;

        const h, l = bit64.pair(n);

        if(s >= 32) then
            return bit64.pair(lshift(l, s - 32), 0);
        end;

        return bit64.pair(
            bor(lshift(h, s), rshift(l, 32 - s)),
            lshift(l, s)
        );
    end;

    function bit64.rshift(n, d)
        n = uint64(n);

        const s = shift(d);

        if(s <= 0) then
            return normalize(n);
        end;

        if(s >= 64) then
            return bit64.zero;
        end;

        const h, l = bit64.pair(n);

        if(s >= 32) then
            return bit64.pair(0, rshift(h, s - 32));
        end;

        return bit64.pair(
            rshift(h, s),
            bor(rshift(l, s), lshift(h, 32 - s))
        );
    end;

    function bit64.lrotate(n, d)
        n = uint64(n);

        const s = shift(d) % 64;

        if(s == 0) then
            return normalize(n);
        end;

        return bit64.bor(
            bit64.lshift(n, s),
            bit64.rshift(n, 64 - s)
        );
    end;

    function bit64.rrotate(n, d)
        n = uint64(n);

        const s = shift(d) % 64;

        if(s == 0) then
            return normalize(n);
        end;

        return bit64.bor(
            bit64.rshift(n, s),
            bit64.lshift(n, 64 - s)
        );
    end;

    function bit64.countlz(n)
        n = uint64(n);

        const h, l = bit64.pair(n);

        if(h == 0) then
            return 32 + countlz(l);
        end;

        return countlz(h);
    end;

    function bit64.countrz(n)
        n = uint64(n);

        const h, l = bit64.pair(n);

        if(l == 0) then
            return 32 + countrz(h);
        end;

        return countrz(l);
    end;

    --- comparison ---

    function bit64.eq(n, x)
        n, x = uint64(n), uint64(x);

        return n.x == x.x and n.y == x.y and n.z == x.z;
    end;

    function bit64.lt(n, x)
        n, x = uint64(n), uint64(x);

        const nh, nl = bit64.pair(n);
        const xh, xl = bit64.pair(x);

        if(nh ~= xh) then
            return nh < xh;
        end;

        return nl < xl;
    end;

    function bit64.le(n, x)
        return bit64.lt(n, x) or bit64.eq(n, x);
    end;

    --- conversion ---

    function bit64.hex(n)
        return format('0x%08X%08X', bit64.pair(uint64(n)));
    end;

    function bit64.decimal(n)
        local h, l = bit64.pair(uint64(n));

        if(h == 0 and l == 0) then
            return '0';
        end;

        local o, d, c = '', {}, 0;
        while(h ~= 0 or l ~= 0) do
            const q = h //      10;
            const r = h  %      10;
            const t = r  * m32 + l;

            h, l = q, t // 10;

            c    += 1;
            d[c] = t % 10;
        end;

        for i = c, 1, -1 do
            o ..= char(48 + d[i]);
        end;

        return o;
    end;
end;

---- exports ----

return(bit64);
