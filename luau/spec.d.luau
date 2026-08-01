---- types ----

export type color  = vector | { r: number, g: number, b: number }

export type uint64 =                                      vector;
export type uint32 =                                      number;

export type uint   =                           (uint64 | uint32);

---- classes ----

export type WebsocketClient = {
    --- constants ---

    Url: string;

    --- methods ---

    --[=[
        Sends a message to the websocket server.

        @param this WebsocketClient
        @param message string
        @param binary boolean?
    ]=]
    Send: (this: WebsocketClient, message: string, binary: boolean?) -> ();

    --[=[
        Closes the websocket connection.

        @param this WebsocketClient
    ]=]
    Disconnect: (this: WebsocketClient) -> ();

    --- events ---

    --[=[
        Fired when the websocket receives a message.

        ```luau
        local ws: WebsocketClient = websocket.connect('ws://localhost:8080');

        ws:DataReceived(function(message: string)
            print(`Received message: {message}`);
        end);
        ```

        @param this WebsocketClient
        @param message string
    ]=]
    DataReceived: (this: WebsocketClient, message: string) -> ();
}

export type handle = {
    --[=[
        Writes `content` to the handle.

        @param this handle
        @param content string
        @return handle
    ]=]
    write:     (this: handle, content: string) -> (handle);

    --[=[
        Appends `content` to the handle.

        @param this handle
        @param content string
        @return handle
    ]=]
    append:    (this: handle, content: string) -> (handle);

    --[=[
        Reads everything from the handle.

        @param this handle
        @return string
    ]=]
    read:      (this: handle)                  -> (string);

    --[=[
        Returns the size of the handle's contents.

        @param this handle
        @return number
    ]=]
    size:      (this: handle)                  -> (number);

    --[=[
        Loads the handle's contents as a Luau file, and returns the callback.

        @param this handle
        @return function
    ]=]
    load:      (this: handle)                  -> ((...any) -> (...any));

    --[=[
        Closes the handle.

        @param this handle
        @return nil
    ]=]
    close:     (this: handle)                  -> ();

    --[=[
        Returns a unix timestamp of when the handle's contents were last modified.

        @param this handle
        @return number
    ]=]
    timestamp: (this: handle)                  -> (number);

    path: string;
}

declare class Doodle
    ZIndex:                                                         number
    Visible:                                                       boolean
    Color:                                                           color
    Opacity:                                                        number
    Transparency:                                                   number

    function Remove(self): nil
end;

declare class Circle extends Doodle
    Filled:                                                       boolean
    Thickness:                                                     number

    NumSides:                                                      number
    Radius:                                                        number
    Position:                                                      vector
end;

declare class Line extends Doodle
    Thickness:                                                     number

    From:                                                          vector
    To:                                                            vector
end;

declare class Text extends Doodle
    OutlineColor:                                                    color
    Text:                                                           string
    Font:                                                           number

    Center:                                                        boolean
    Outline:                                                       boolean

    Size:                                                           number
    TextBounds:                                                     vector
    Position:                                                       vector
end;

declare class Square extends Doodle
    Thickness:                                                      number
    Rounding:                                                       number

    Filled:                                                        boolean

    Position:                                                       vector
    Size:                                                           vector
end;

declare class Image extends Doodle
    Data:                                                           string
    Gif:                                                           boolean
    Delay:                                                          number
    Url:                                                            string

    Size:                                                           vector
    ImageSize:                                                      vector
    Position:                                                       vector
end;

declare class Triangle extends Doodle
    Filled:                                                       boolean
    Thickness:                                                     number

    PointA:                                                        vector
    PointB:                                                        vector
    PointC:                                                        vector
end;

declare class Quad extends Triangle
    PointD:                                                        vector
end;

declare class Instance
    Name:                                                           string
    ClassName:                                                      string
    Parent:                                                      Instance?

    function IsA(self, class: string):                             boolean
    function Destroy(self):                                            nil
    function GetChildren(self):                               { Instance }
    function FindFirstChild(self, name: string):                 Instance?
    function WaitForChild(self, name: string, timeout: number?): Instance?
end;

---- libraries ----

--- Scheduling, and managing tasks.
export type task = {
    --- functions ---

    --[=[
        Schedules a task to run in the next frame.

        @param f function | thread
        @return thread
    ]=]
    defer: (f: () -> ()) -> thread;

    --[=[
        Schedules a task to run after the given delay in seconds.

        @param f function | thread
        @param d number
        @return thread
    ]=]
    delay: (f: () -> (), d: number) -> thread;

    --[=[
        Immediately executes the provided function/thread `f` in a separate thread without yielding the current thread.

        @param f function | thread
        @return thread
    ]=]
    spawn: (f: () -> ()) -> thread;

    --[=[
        Cancels a scheduled task.

        @param t thread
        @return nil
    ]=]
    cancel: (t: thread) -> nil;

    --[=[
        Yields the current thread for the given delay in seconds.
        Returns the actual time waited in seconds.

        @param d number
        @return number
    ]=]
    wait: (d: number) -> number;
}

--- Namespace for reading, writing, and manipulating memory.
export type memory = {
    --- constants ---

    base: number;

    --- functions ---

    --[=[
        Reads a signed 8-bit `[0, 127]` integer.

        ```luau
        local value: number = memory.readi8(game, 0x8);
        print(value); -- volatile, but probably a number.
        ```

        @param a userdata | number
        @param o number?
        @return number
    ]=]
    readi8:
        & (a: Instance, o: number) -> number
        & (a: number)              -> number;

    --[=[
        Reads an unsigned 8-bit `[0, 255]` integer.

        @param a userdata | number
        @param o number?
        @return number
    ]=]
    readu8:
        & (a: Instance, o: number) -> number
        & (a: number)              -> number;

    --[=[
        Reads a signed 16-bit `[-32768, 32767]` integer.

        @param a userdata | number
        @param o number?
        @return number
    ]=]
    readi16:
        & (a: Instance, o: number) -> number
        & (a: number)              -> number;

    --[=[
        Reads an unsigned 16-bit `[0, 65535]` integer.

        @param a userdata | number
        @param o number?
        @return number
    ]=]
    readu16:
        & (a: Instance, o: number) -> number
        & (a: number)              -> number;

    --[=[
        Reads a signed 32-bit `[-2147483648, 2147483647]` integer.

        @param a userdata | number
        @param o number?
        @return number
    ]=]
    readi32:
        & (a: Instance, o: number) -> number
        & (a: number)              -> number;

    --[=[
        Reads an unsigned 32-bit `[0, 4294967295]` integer.

        @param a userdata | number
        @param o number?
        @return number
    ]=]
    readu32:
        & (a: Instance, o: number) -> number
        & (a: number)              -> number;

    --[=[
        Reads an unsigned 64-bit `[0, 18446744073709551615]` integer.

        @param a userdata | number
        @param o number?
        @return uint64
    ]=]
    readu64:
        & (a: Instance, o: number, s: number, d: number) -> number
        & (a: number, s: number, d: number)              -> number;

    --[=[
        Reads a range of bits from the given address, starting at the given bit offset `s`, and ending at the given bit offset `d`.

        @param a userdata | number
        @param o number?
        @param s number
        @param d number
        @return uint64
    ]=]
    readbits:
        & (a: Instance, o: number, s: number, d: number) -> number
        & (a: number, s: number, d: number)              -> number;

    --[=[
        Reads a string from the given address until the null terminator.

        @param a userdata | number
        @param o number?
        @return string
    ]=]
    readstring:
        & (a: Instance, o: number) -> string
        & (a: number)              -> string;

    --[=[
        Reads a 3-dimensional vector from the given address.

        @param a userdata | number
        @param o number?
        @return vector
    ]=]
    readvector:
        & (a: Instance, o: number) -> vector
        & (a: number)              -> vector;

    --[=[
        Reads a buffer from the given address, sized to the given size `s`.

        @param a userdata | number
        @param o number?
        @param s number
        @return buffer
    ]=]
    readbuffer:
        & (a: Instance, o: number, s: number) -> buffer
        & (a: number, s: number)              -> buffer;

    --[=[
        Reads the RTTI (Run-Time Type Information) from the given address.

        @param a userdata | number
        @param o number?
        @return string
    ]=]
    rtti:
        & (a: Instance, o: number) -> string
        & (a: number) -> string;

    --[=[
        Writes a signed 8-bit `[0, 127]` integer to the given address.

        @param a userdata | number
        @param o number?
        @param v number
        @return nil
    ]=]
    writei8:
        & (a: Instance, o: number, v: number) -> nil
        & (a: number, v: number)              -> nil;

    --[=[
        Writes an unsigned 8-bit `[0, 255]` integer to the given address

        @param a userdata | number
        @param o number?
        @param v number
        @return nil
    ]=]
    writeu8:
        & (a: Instance, o: number, v: number) -> nil
        & (a: number, v: number)              -> nil;

    --[=[
        Writes a signed 16-bit `[-32768, 32767]` integer to the given address.

        @param a userdata | number
        @param o number?
        @param v number
        @return nil
    ]=]
    writei16:
        & (a: Instance, o: number, v: number) -> nil
        & (a: number, v: number)              -> nil;

    --[=[
        Writes an unsigned 16-bit `[0, 65535]` integer to the given address.

        @param a userdata | number
        @param o number?
        @param v number
        @return nil
    ]=]
    writeu16:
        & (a: Instance, o: number, v: number) -> nil
        & (a: number, v: number)              -> nil;

    --[=[
        Writes a signed 32-bit `[-2147483648, 2147483647]` integer to the given address.

        @param a userdata | number
        @param o number?
        @param v number
        @return nil
    ]=]
    writei32:
        & (a: Instance, o: number, v: number) -> nil
        & (a: number, v: number)              -> nil;

    --[=[
        Writes an unsigned 32-bit `[0, 4294967295]` integer to the given address.

        @param a userdata | number
        @param o number?
        @param v number
        @return nil
    ]=]
    writeu32:
        & (a: Instance, o: number, v: number) -> nil
        & (a: number, v: number)              -> nil;

    --[=[
        Writes an unsigned 64-bit `[0, 18446744073709551615]` integer to the given address.

        @param a userdata | number
        @param o number?
        @param v uint64
        @return nil
    ]=]
    writeu64:
        & (a: Instance, o: number, v: uint64) -> nil
        & (a: number, v: uint64)              -> nil;

    --[=[
        Writes a range of bits to the given address, starting at the given bit offset `s`, and ending at the given bit offset `d`.

        @param a userdata | number
        @param o number?
        @param s number
        @param d number
        @param v uint64
        @return nil
    ]=]
    writebits:
        & (a: Instance, o: number, s: number, d: number, v: number) -> nil
        & (a: number, s: number, d: number, v: number)              -> nil;

    --[=[
        Writes a string to the given address, and appends a null terminator.

        @param a userdata | number
        @param o number?
        @param v string
        @return nil
    ]=]
    writestring:
        & (a: Instance, o: number, v: string) -> nil
        & (a: number, v: string)              -> nil;

    --[=[
        Writes a 3-dimensional vector to the given address.

        @param a userdata | number
        @param o number?
        @param v vector
        @return nil
    ]=]
    writevector:
        & (a: Instance, o: number, v: vector) -> nil
        & (a: number, v: vector)              -> nil;

    --[=[
        Writes a buffer to the given address.

        @param a userdata | number
        @param o number?
        @param v buffer
        @return nil
    ]=]
    writebuffer:
        & (a: Instance, o: number, v: buffer) -> nil
        & (a: number, v: buffer)              -> nil;
}

--- A 64-bit unsigned integer type, with support for arithmetic, and bitwise operations, as well as conversion, and comparisons.
export type bit64 = {
    --- constructors ---

    --[=[
        Constructs a new `uint64` from the given high and low values.
        A `uint64` can also be provided to be converted into high, and low values.

        @param h number | uint64
        @param l number
        @return uint64 | (number, number)

        ```luau
        local paired: uint64 = bit64.pair(0x00000000, 0xFFFFFFFF)
        local
            h: number,
            l: number
        = bit64.pair(paired)

        print(h, l) -- 0, 4294967295
        ```
    ]=]
    pair:
        & ((h: number, l: number)           ->         (uint64))
        & ((uint64)                         -> (number, number));

    --[=[
        Constructs a new `uint64` from the given buffer.
        A `uint64` can also be provided to be converted/written into a buffer.

        ```luau
        local block: buffer = bit64.buffer(5124);
        print(bit64.decimal(bit64.buffer(block))); -- 5124
        ```

        @param b buffer
        @param o number?
        @return uint64 | buffer?
    ]=]
    buffer:
        & ((b: buffer, o: number?)          ->         (uint64))
        & ((uint64, b: buffer?, o: number?) ->        (buffer?));

    --[=[
        Constructs a new `uint64` from the given double value.
        A `uint64` can also be provided to be converted into a double value.

        @param n number | uint64
        @return uint64 | number
    ]=]
    double:
        & ((n: number)                      ->         (uint64))
        & ((uint64)                         ->         (number));

    --- constants ---

    zero:                                                 uint64;

    --- arithmetic ---

    --[=[
        Performs addition on the given values.

        ```luau
        local a: uint64 = bit64.double(512);
        local b: uint64 = bit64.double(1024);
        local c: uint64 = bit64.add(a, b);

        print(bit64.decimal(c)); -- 1536
        ```

        @param n uint64 | number
        @param x uint64 | number
        @return uint64
    ]=]
    add: (
        n:                                                  uint,
        x:                                                  uint
    ) -> uint64;

    --[=[
        Performs subtraction on the given values.

        ```luau
        local a: uint64 = bit64.double(1024);
        local b: uint64 = bit64.double(512);
        local c: uint64 = bit64.sub(a, b);

        print(bit64.decimal(c)); -- 512
        ```

        @param n uint64 | number
        @param x uint64 | number
        @return uint64
    ]=]
    sub: (
        n:                                                  uint,
        x:                                                  uint
    ) -> uint64;

    --[=[
        Performs multiplication on the given values.

        ```luau
        local a: uint64 = bit64.double(512);
        local b: uint64 = bit64.double(512);
        local c: uint64 = bit64.mul(a, b);

        print(bit64.decimal(c)); -- 262144
        ```

        @param n uint64 | number
        @param x uint64 | number
        @return uint64
    ]=]
    mul: (
        n:                                                  uint,
        x:                                                  uint
    ) -> uint64;

    --[=[
        Performs division on the given values.

        ```luau
        local a: uint64 = bit64.double(1024);
        local b: uint64 = bit64.double(512);
        local c: uint64 = bit64.div(a, b);

        print(bit64.decimal(c)); -- 2
        ```

        @param n uint64 | number
        @param x uint64 | number
        @return uint64
    ]=]
    div: (
        n:                                                  uint,
        x:                                                  uint
    ) -> uint64;

    --[=[
        Performs modulo on the given values.

        ```luau
        local a: uint64 = bit64.double(1025);
        local b: uint64 = bit64.double(512);
        local c: uint64 = bit64.mod(a, b);

        print(bit64.decimal(c)); -- 1
        ```

        @param n uint64 | number
        @param x uint64 | number
        @return uint64
    ]=]
    mod: (
        n:                                                  uint,
        x:                                                  uint
    ) -> uint64;

    --[=[
        Performs exponentiation on the given values.

        ```lua
        local a: uint64 = bit64.double(2);
        local b: uint64 = bit64.double(10);
        local c: uint64 = bit64.pow(a, b);

        print(bit64.decimal(c)); -- 1024
        ```

        @param n uint64 | number
        @param x uint64 | number
        @return uint64
    ]=]
    pow: (
        n:                                                  uint,
        x:                                                  uint
    ) -> uint64;

    --- bitwise ---

    --[=[
        Bitwise AND across the given values.

        @param n uint64 | number
        @param x uint64 | number
        @return uint64
    ]=]
    band: (
        ...                                                 uint
    ) -> uint64;

    --[=[
        Bitwise OR across the given values.

        @param n uint64 | number
        @param x uint64 | number
        @return uint64
    ]=]
    bor: (
        ...                                                 uint
    ) -> uint64;

    --[=[
        Bitwise XOR across the given values.

        @param n uint64 | number
        @param x uint64 | number
        @return uint64
    ]=]
    bxor: (
        ...                                                 uint
    ) -> uint64;

    --[=[
        Bitwise NOT.

        @param n uint64 | number
        @return uint64
    ]=]
    bnot: (
        n:                                                  uint
    ) -> uint64;

    --[=[
        Left shift by `n` bits.
        Clamped to `[0, 63]`.

        @param n uint64 | number
        @param d number
        @return uint64
    ]=]
    lshift: (
        n:                                                  uint,
        d:                                                  uint
    ) -> uint64;

    --[=[
        Right shift by `n` bits.
        Clamped to `[0, 63]`.

        @param n uint64 | number
        @param d number
        @return uint64
    ]=]
    rshift: (
        n:                                                  uint,
        d:                                                  uint
    ) -> uint64;

    --[=[
        Rotate left by `n` bits.

        @param n uint64 | number
        @param d number
        @return uint64
    ]=]
    lrotate: (
        n:                                                  uint,
        d:                                                  uint
    ) -> uint64;

    --[=[
        Rotate right by `n` bits.

        @param n uint64 | number
        @param d number
        @return uint64
    ]=]
    rrotate: (
        n:                                                  uint,
        d:                                                  uint
    ) -> uint64;

    --[=[
        Counts the number of leading zeros in the given value.

        @param n uint64 | number
        @return uint64
    ]=]
    countlz: (
        n:                                                  uint
    ) -> uint64;

    --[=[
        Counts the number of trailing zeros in the given value.

        @param n uint64 | number
        @return uint64
    ]=]
    countrz: (
        n:                                                  uint
    ) -> uint64;

    --- comparison ---

    --[=[
        Returns a boolean based on whether the two values are equal.

        @param n uint64 | number
        @param x uint64 | number
        @return boolean
    ]=]
    eq: (
        n:                                                  uint,
        x:                                                  uint
    ) -> boolean;

    --[=[
        Returns a boolean based on whether the first value is less than the second.

        @param n uint64 | number
        @param x uint64 | number
        @return boolean
    ]=]
    lt: (
        n:                                                  uint,
        x:                                                  uint
    ) -> boolean;

    --[=[
        Returns a boolean based on whether the first value is less or equal than the second.

        @param n uint64 | number
        @param x uint64 | number
        @return boolean
    ]=]
    le: (
        n:                                                  uint,
        x:                                                  uint
    ) -> boolean;

    --- conversion ---

    --[=[
        Converts the given value to a hexadecimal string.

        @param n uint64 | number
        @return string
    ]=]
    hex: (
        n:                                                  uint
    ) -> string;

    --[=[
        Converts the given value to a decimal string.

        @param n uint64 | number
        @return string
    ]=]
    decimal: (
        n:                                                  uint
    ) -> string;
};

--- File system operations, including reading, writing, creating, and deleting files and folders.
export type fs = {
    --[=[
        Opens a file at the given path, and returns a handle to it.

        @param path string
        @return handle
    ]=]
    open:      (path: string)                  -> (handle);

    --[=[
        Reads the file at the given path, and returns its contents.

        @param path string
        @return string
    ]=]
    read:      (path: string)                  -> (string);

    --[=[
        Returns a list of every entry within the given path.

        @param path string
        @return { string }
    ]=]
    entries:   (path: string)                  -> { string };

    --[=[
        Returns a boolean based on whether the file at the given path exists.

        @param path string
        @return boolean
    ]=]
    file:      (path: string)                  -> (boolean);

    --[=[
        Returns a boolean based on whether the folder at the given path exists.

        @param path string
        @return boolean
    ]=]
    folder:    (path: string)                  -> (boolean);

    --[=[
        Writes the file at the given path with the provided content.

        @param path string
        @param content string
        @return nil
    ]=]
    write:     (path: string, content: string) -> ();

    --[=[
        Loads a Luau file at the given path, and returns the callback.

        @param path string
        @return function
    ]=]
    load:      (path: string)                  -> ((...any) -> (...any));

    --[=[
        Creates a folder at the given path.

        @param path string
        @return nil
    ]=]
    make:      (path: string)                  -> ();

    --[=[
        Removes a folder at the given path.

        @param path string
        @return nil
    ]=]
    remove:    (path: string)                  -> ();

    --[=[
        Deletes a file at the given path.

        @param path string
        @return nil
    ]=]
    delete:    (path: string)                  -> ();

    --[=[
        Returns a unix timestamp of when the file at the given path was last modified.

        @param path string
        @return number
    ]=]
    timestamp: (path: string)                  -> (number);
}

---- declarations ----

declare bit64:   bit64;
declare memory: memory;
declare task:     task;
declare fs:         fs;

declare WebsocketClient: {
    --[=[
        Connects to a websocket server at the given URL, and returns a `WebsocketClient` instance.

        @yields
        @param url string
        @return WebsocketClient
    ]=]
    new: (url: string) -> WebsocketClient
};

--- Retained-mode graphics.
declare Drawing: {
    --[=[
        Registers a font with the given name, size, and data.

        ```luau
        Drawing.RegisterFont("BebasNeue", 20, link_to_font);
        ```

        @param name string
        @param size number
        @param data string
    ]=]
    RegisterFont: (name: string, size: number, data: string) -> (),

    --- Clears all drawings from the screen.
    clear: () -> (),

    --[=[
        Creates a new drawing of the given type, and returns it.
        Note: Deprecated in favor of using type constructors, i.e: `Line.new()`, or `Circle.new()`.

        @param type string
        @return Drawing
    ]=]
    new: (("Line")     -> Line)     &
         (("Text")     -> Text)     &
         (("Image")    -> Image)    &
         (("Circle")   -> Circle)   &
         (("Square")   -> Square)   &
         (("Triangle") -> Triangle) &
         (("Quad")     -> Quad)
}

--- Immediate-mode graphics.
declare DrawingImmediate: {
    Line: (
        From:       vector,
        To:         vector,
        Color:       color,
        Opacity:    number,
        Thickness:  number
    ) -> ();

    Circle: (
        Center:     vector,
        Radius:     number,
        Color:       color,
        Opacity:    number,
        Segments:   number,
        Thickness:  number
    ) -> ();

    FilledCircle: (
        Center:     vector,
        Radius:     number,
        Color:       color,
        Opacity:    number,
        Segments:   number
    ) -> ();

    Triangle: (
        PointA:     vector,
        PointB:     vector,
        PointC:     vector,
        Color:       color,
        Opacity:    number,
        Thickness:  number
    ) -> ();

    FilledTriangle: (
        PointA:     vector,
        PointB:     vector,
        PointC:     vector,
        Color:       color,
        Opacity:    number
    ) -> ();

    Rectangle: (
        Position:   vector,
        Size:       vector,
        Color:       color,
        Opacity:    number,
        Rounding:   number,
        Thickness:  number
    ) -> ();

    FilledRectangle: (
        Position:   vector,
        Size:       vector,
        Color:       color,
        Opacity:    number,
        Rounding:   number
    ) -> ();

    Quad: (
        PointA:     vector,
        PointB:     vector,
        PointC:     vector,
        PointD:     vector,
        Color:       color,
        Opacity:    number,
        Thickness:  number
    ) -> ();

    FilledQuad: (
        PointA:     vector,
        PointB:     vector,
        PointC:     vector,
        PointD:     vector,
        Color:       color,
        Opacity:    number
    ) -> ();

    Polyline: (
        Points: { vector },
        Color:       color,
        Opacity:    number,
        Thickness:  number
    ) -> ();

    Text: (
        Position:   vector,
        Size:       number,
        Color:       color,
        Opacity:    number,
        Text:       string,
        Centered:  boolean,
        Font:       string
    ) -> ();

    OutlinedText: (
        Position:   vector,
        Size:       number,
        Color:       color,
        Opacity:    number,
        Text:       string,
        Centered:  boolean,
        Font:       string
    ) -> ();

    GetTextBounds: (
        Font:       string,
        Size:       number,
        Text:       string
    ) -> (vector);
}

declare crypt: {
    --[=[
        Generates cryptographically secure random bytes of the specified size.

        @param size number
        @return string
    ]=]
    random: (size: number) -> string;

    --[=[
        Generates deterministic random bytes of the specified size using the given seed.

        @param seed number
        @param size number
        @return string
    ]=]
    random_deterministic: (seed: number, size: number) -> string;

    --- Namespace dedicated to hashing algorithms.
    hash: {
        --[=[
            Generates a SHA-256 hash of the given data.

            @param data string
            @return string
        ]=]
        sha256: (data: string) -> string;

        --[=[
            Generates a SHA-512 hash of the given data.

            @param data string
            @return string
        ]=]
        sha512: (data: string) -> string;

        --[=[
            Generates a BLAKE2b hash of the given data.

            @param data string
            @return string
        ]=]
        blake2b: (data: string) -> string;
    };

    --[=[
        Generates a password hash of the given password and salt.

        @param password string
        @param salt string?
        @return string
    ]=]
    pwhash: (password: string, salt: string?) -> string;

    --[=[
        Generates a password hash of the given password and salt, and returns both the hash and salt.

        @param password string
        @return (string, string)
    ]=]
    pwhash_str: (password: string) -> (string, string);

    --[=[
        Verifies a password against a given hash.

        @param hash string
        @param password string
        @return boolean
    ]=]
    pwhash_str_verify: (hash: string, password: string) -> boolean;

    --- Namespace dedicated to symmetric encryption algorithms.
    secretbox: {
        --[=[
            Encrypts a message using a secret key and nonce.

            @param message string
            @param nonce string
            @param key string
            @return string
        ]=]
        seal: (message: string, nonce: string, key: string) -> string;

        --[=[
            Decrypts a sealed ciphertext using a secret key and nonce.

            @param cipher string
            @param nonce string
            @param key string
            @return string
        ]=]
        open: (cipher: string,  nonce: string, key: string) -> string;
    };

    --- Namespace dedicated to asymmetric encryption algorithms.
    aead: {
        --[=[
            Encrypts a message with authenticated associated data using a key and nonce.

            @param message string
            @param aad string
            @param nonce string
            @param key string
            @return string
        ]=]
        encrypt: (message: string, aad: string, nonce: string, key: string) -> string;

        --[=[
            Decrypts a ciphertext with authenticated associated data using a key and nonce.

            @param cipher string
            @param aad string
            @param nonce string
            @param key string
            @return string
        ]=]
        decrypt: (cipher: string, aad: string, nonce: string, key: string) -> string;
    };

    --- Namespace dedicated to public-key encryption algorithms.
    box: {
        --[=[
            Generates a public/private key pair for public-key encryption.

            @return (string, string)
        ]=]
        keypair: () -> (string, string);

        --[=[
            Encrypts a message for a recipient using their public key and the sender's private key.

            @param message string
            @param nonce string
            @param public string
            @param private string
            @return string
        ]=]
        encrypt: (message: string, nonce: string, public: string, private: string) -> string;

        --[=[
            Decrypts a ciphertext using the sender's public key and the recipient's private key.

            @param cipher string
            @param nonce string
            @param public string
            @param private string
            @return string
        ]=]
        decrypt: (cipher: string, nonce: string, public: string, private: string) -> string;

        --[=[
            Anonymously encrypts a message for a recipient using only their public key.

            @param message string
            @param public string
            @return string
        ]=]
        seal: (message: string, public: string) -> string;

        --[=[
            Decrypts an anonymously sealed ciphertext using the recipient's public and secret keys.

            @param cipher string
            @param public string
            @param secret string
            @return string
        ]=]
        open: (cipher: string, public: string, secret: string) -> string;

        --[=[
            Precomputes a shared key from a public and private key pair for use with secretbox.

            @param public string
            @param private string
            @return string
        ]=]
        beforenm: (public: string, private: string) -> string;
    };

    --- Namespace dedicated to digital signature algorithms.
    sign: {
        --[=[
            Generates a public/private key pair for digital signatures.

            @return (string, string)
        ]=]
        keypair: () -> (string, string);

        --[=[
            Signs a message with a private key, returning the signed message.

            @param message string
            @param private string
            @return string
        ]=]
        sign: (message: string, private: string) -> string;

        --[=[
            Verifies and opens a signed message using a public key, returning the original message.

            @param message string
            @param public string
            @return string
        ]=]
        open: (message: string, public: string) -> string;

        --[=[
            Creates a detached signature for a message using a private key.

            @param message string
            @param private string
            @return string
        ]=]
        detached: (message: string, private: string) -> string;

        --[=[
            Verifies a detached signature against a message using a public key.

            @param signature string
            @param message string
            @param public string
            @return boolean
        ]=]
        verify_detached: (signature: string, message: string, public: string) -> boolean;

        --[=[
            Verifies a signature against a message using a public key.

            @param message string
            @param signature string
            @param public string
            @return boolean
        ]=]
        verify: (message: string, signature: string, public: string) -> boolean;
    };

    --- Namespace dedicated to base64 encoding and decoding.
    base64: {
        --[=[
            Encodes binary data as a base64 string.

            @param data string
            @return string
        ]=]
        encode: (data: string) -> string;

        --[=[
            Decodes a base64 string into binary data.

            @param data string
            @return string
        ]=]
        decode: (data: string) -> string;
    };

    --- Namespace dedicated to JSON encoding and decoding.
    json: {
        --[=[
            Encodes a value as a JSON string.

            @param data any
            @return string
        ]=]
        encode: (data: any) -> string;

        --[=[
            Decodes a JSON string into a Luau value.

            @param data string
            @return any
        ]=]
        decode: (data: string) -> any;
    };

    --- Namespace dedicated to hexadecimal encoding and decoding.
    hexadecimal: {
        --[=[
            Encodes binary data as a hexadecimal string.

            @param data string
            @return string
        ]=]
        encode: (data: string) -> string;

        --[=[
            Decodes a hexadecimal string into binary data.

            @param data string
            @return string
        ]=]
        decode: (data: string) -> string;
    };

    --- Namespace dedicated to HMAC key derivation.
    hmac: {
        --[=[
            Computes an HMAC-SHA-256 digest of the given data using the provided key.

            @param key string
            @param data string
            @return string
        ]=]
        sha256: (key: string, data: string) -> string;

        --[=[
            Computes an HMAC-SHA-512 digest of the given data using the provided key.

            @param key string
            @param data string
            @return string
        ]=]
        sha512: (key: string, data: string) -> string;
    };

    --- Namespace dedicated to HKDF key derivation.
    hkdf: {
        --[=[
            Derives a key using HKDF-SHA-256 from the given key material, salt, and info.

            @param key string
            @param salt string
            @param info string
            @return string
        ]=]
        sha256: (key: string, salt: string, info: string) -> string;
    };
};

--- Returns a boolean based on whether Roblox is focused.
declare function isrbxactive(): boolean

--- Returns the window size.
declare function getscreendimensions(): vector

--- Override/creates a file at the given path with `content`. Deprecated in favor of `fs.write`.
@deprecated declare function writefile(path: string, content: string): ()

--- Reads the file at the given path and returns its content. Deprecated in favor of `fs.read`.
@deprecated declare function readfile(path: string): string

--- Appends `content` to the file at the given path. Deprecated in favor of `fs.write`.
@deprecated declare function appendfile(path: string, content: string): ()

--- Returns a boolean based on whether the file at the given path exists. Deprecated in favor of `fs.file`.
@deprecated declare function isfile(path: string): boolean

--- Deletes the file at the given path. Deprecated in favor of `fs.delete`.
@deprecated declare function delfile(path: string): ()

--- Creates a folder at the given path. Deprecated in favor of `fs.make`.
@deprecated declare function makefolder(path: string): ()

--- Returns a boolean based on whether the folder at the given path exists. Deprecated in favor of `fs.folder`.
@deprecated declare function isfolder(path: string): boolean

--- Deletes the folder at the given path. Deprecated in favor of `fs.remove`.
@deprecated declare function delfolder(path: string): ()

--- Returns a list of files in the folder at the given path. Deprecated in favor of `fs.entries`.
@deprecated declare function listfiles(path: string): { string }

--- Loads a Luau file at the given path, and returns the callback. Deprecated in favor of `fs.load`.
@deprecated declare function loadfile(path: string): () -> ()

--[=[
    Moves the mouse cursor relative to its current position.

    @param x number
    @param y number
]=]
declare function mousemoverel(x: number, y: number): ()

--[=[
    Moves the mouse cursor to the given absolute position.

    @param x number
    @param y number
]=]
declare function mousemoveabs(x: number, y: number): ()

--[=[
    Returns the current mouse position.

    @return vector
]=]
declare function getmouseposition(): vector

--[=[
    Linearly interpolates the mouse cursor from `origin` to `target` at the given `speed`.

    @param origin vector
    @param target vector
    @param speed number

    ```luau
    local origin:      vector = getmouseposition();
    local target:      vector = vector.create(512, 512);
    local sensitivity: number = 1;
    local speed:       number = 0.5;

    local result = smoothmouse_linear(origin, target, sensitivity, speed);
    print(result)
    ```
]=]
declare function smoothmouse_linear(origin: vector, target: vector, sensitivity: number, speed: number): vector

--[=[
    Exponentially interpolates the mouse cursor from `origin` to `target` at the given `speed`.

    @param origin vector
    @param target vector
    @param speed number

    ```luau
    local origin: vector = getmouseposition();
    local target: vector = vector.create(512, 512);
    local speed:  number = 1;

    local result = smoothmouse_exponential(origin, target, speed);
    print(result)
    ```
]=]
declare function smoothmouse_exponential(origin: vector, target: vector, speed: number): vector

--- Simulates a left mouse button click.
declare function mouse1click(): ()

--- Simulates the press of the left mouse button.
declare function mouse1press(): ()

--- Simulates the release of the left mouse button.
declare function mouse1release(): ()

--- Simulates a right mouse button click.
declare function mouse2click(): ()

--- Simulates the press of the right mouse button.
declare function mouse2press(): ()

--- Simulates the press of the right mouse button
declare function mouse2release(): ()

--[=[
    Simulates scrolling by `delta`.

    @param delta number
]=]
declare function mousescroll(delta: number): ()

--[=[
    Simulates a key press of the given key code.

    @param key number
]=]
declare function keypress(key: number): ()

--[=[
    Simulates a key release of the given key code.

    @param key number
]=]
declare function keyrelease(key: number): ()

--[=[
    Returns an array of all currently pressed keys.

    @return { string }
]=]
declare function getpressedkeys(): { string }

--[=[
    Returns the last pressed key.

    @return string
]=]
declare function getpressedkey(): string

--[=[
    Returns a boolean based on whether the right mouse button is currently pressed.

    @return boolean
]=]
declare function isrightpressed(): boolean

--[=[
    Returns a boolean based on whether the left mouse button is currently pressed.

    @return boolean
]=]
declare function isleftpressed(): string

--[=[
    Returns a boolean based on whether the right mouse button was clicked.

    @return boolean
]=]
declare function isrightclicked(): boolean

--[=[
    Returns a boolean based on whether the left mouse button was clicked.

    @return boolean
]=]
declare function isleftclicked(): boolean

--[=[
    Sets the clipboard to the given text.

    @param text string
]=]
declare function setclipboard(text: string): ()

--[=[
    Converts the pointer into an Instance.

    @param source number
    @return Instance
]=]
declare function pointer_to_userdata(source: number): Instance

--- Clears user-created local data, returns true if successful.
declare function clear_local_data(): boolean

--- Clears user-created model data, returns true if successful.
declare function clear_model_data(): boolean

--[=[
    Overrides local data with the given configuration.
    - `LocalPlayer: Instance`
    - `DisplayName: string`
    - `Username: string`
    - `UserId: number`
    - `Character: Instance`
    - `Team: Instance`
    - `RootPart: Instance`
    - `LeftFoot: Instance`
    - `Head: Instance`
    - `LowerTorso: Instance`
    - `Tool: Instance`
    - `Humanoid: Instance`
    - `Health: number`
    - `MaxHealth: number`
    - `RigType: number` 0 = R6, 1 = R15

    @param Configuration table
    @return boolean
]=]
declare function override_local_data(
    configuration: {
        LocalPlayer: Instance,
        DisplayName:   string,
        Username:      string,
        UserId:        number,
        Character:   Instance,
        Team:        Instance,
        RootPart:    Instance,
        LeftFoot:    Instance,
        Head:        Instance,
        LowerTorso:  Instance,
        Tool:        Instance,
        Humanoid:    Instance,
        Health:        number,
        MaxHealth:     number,
        RigType:       number
    }
): boolean

--[=[
    Adds model data with the given configuration.
    - `Username: string`
    - `DisplayName: string`
    - `UserId: number`
    - `Character: Instance`
    - `PrimaryPart: Instance`
    - `Head: Instance`
    - `LeftLeg: Instance`
    - `LeftArm: Instance`
    - `RightLeg: Instance`
    - `RightArm: Instance`
    - `Torso: Instance`
    - `ToolName: string`
    - `TeamName: string`
    - `BodyHeightScale: number` Height scale multiplier for the character.
    - `RigType: number` 0 = R6, 1 = R15
    - `Whitelisted: boolean`
    - `Archenemies: boolean`
    - `Aimbot_Part: Instance` Part used for the aimbot.
    - `Aimbot_TP_Part: Instance` Part used for the "mouse teleport" feature.
    - `Triggerbot_Part: Instance` Part used for the triggerbot.
    - `Health: number`
    - `MaxHealth: number`

    @param Configuration table
]=]
declare function add_model_data(
    configuration: {
        Username: string,
        DisplayName: string,
        UserId: number,
        Character: Instance,
        PrimaryPart: Instance,
        Head: Instance,
        LeftLeg: Instance,
        LeftArm: Instance,
        RightLeg: Instance,
        RightArm: Instance,
        Torso: Instance,
        ToolName: string,
        TeamName: string,
        BodyHeightScale: number,
        RigType: number,
        Whitelisted: boolean,
        Archenemies: boolean,
        Aimbot_Part: Instance,
        Aimbot_TP_Part: Instance,
        Triggerbot_Part: Instance,
        Health: number,
        MaxHealth: number,
        body_parts_data: { { name: string, part: Instance } },
        full_body_data: { { name: string, part: Instance } }
    }
): boolean

--[=[
    Edits model data with the given configuration.
    - `RigType: number` 0 = R6, 1 = R15
    - `Whitelisted: boolean`
    - `Archenemies: boolean`
    - `Aimbot_Part: Instance` Part used for the aimbot.
    - `Aimbot_TP_Part: Instance` Part used for the "mouse teleport" feature.
    - `Triggerbot_Part: Instance` Part used for the triggerbot.
    - `Health: number`
    - `MaxHealth: number`
    - `BodyHeightScale: number` Height scale multiplier for the character.

    @param Configuration table
    @param key string
]=]
declare function edit_model_data(
    configuration: {
        RigType: number,
        Whitelisted: boolean,
        Archenemies: boolean,
        Aimbot_Part: Instance,
        Aimbot_TP_Part: Instance,
        Triggerbot_Part: Instance,
        Health: number,
        MaxHealth: number,
        BodyHeightScale: number
    }, key: string
): boolean

--[=[
    Removes model data with the given key.

    @param key string
]=]
declare function remove_model_data(key: string): boolean

--[=[
    Returns the current overlay FPS.

    @return number
]=]
declare function get_overlay_fps(): number

--[=[
    Returns a boolean based on whether the menu is currently opened.

    @return boolean
]=]
declare function is_menu_opened(): boolean

--- Blocks inputs from reaching the Roblox window, returns nil.
declare function block_roblox_window(): nil

--[=[
    Queues a function to be executed after the player teleports.

    @param code string
]=]
declare function queue_on_teleport(code: string): nil
