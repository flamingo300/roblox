--!nolint

---- declarations ----

export type handle = {
    write:     (this: handle, content: string) -> (handle);                 -- :: write `content` to the handle.
    read:      (this: handle)                  -> (string);                 -- :: read everything from the handle.
    size:      (this: handle)                  -> (number);                 -- :: get the size of the handle's contents.
    load:      (this: handle)                  -> ((...any) -> (...any));   -- :: load's the file.
    close:     (this: handle)                  -> ();                       -- :: close the handle.
    timestamp: (this: handle)                  -> (number);                 -- :: returns a unix timestamp of when the file was last modified.

    path: string;
}

export type fs = {
    open:      (path: string)                  -> (handle);               -- :: open a file
    read:      (path: string)                  -> (string);               -- :: read the file's contents
    entries:   (path: string)                  -> { string };             -- :: get every entry within the `path`
    file:      (path: string)                  -> (boolean);              -- :: check if the file at `path` exists.
    folder:    (path: string)                  -> (boolean);              -- :: check if the folder at `path` exists.
    write:     (path: string, content: string) -> ();                     -- :: write the file directly
    load:      (path: string)                  -> ((...any) -> (...any)); -- :: load's the file.
    make:      (path: string)                  -> ();                     -- :: create a folder
    remove:    (path: string)                  -> ();                     -- :: remove a folder
    delete:    (path: string)                  -> ();                     -- :: delete a file
    timestamp: (path: string)                  -> (number);               -- :: returns a unix timestamp of when the file was last modified.
}

---- constants ----

const timestamps_path = '.timestamps.txt';

---- utilities ----

const timestamps = {} :: { [string]: number }; do
    --- functions ---

    function timestamps.read(): { [string]: number }
        if(not isfile(timestamps_path)) then
            writefile(timestamps_path, '');

            return {};
        end;

        const entries = {} :: { [string]: number };
        const content = readfile(timestamps_path) :: string;

        for line in content:gmatch('[^\r\n]+') do
            const path, time = line:match('^([^,]+),(%d+)$');
            if(path and time) then
                entries[path] = tonumber(time) :: number;
            end;
        end;

        return(entries);
    end;

    function timestamps.write(entries: { [string]: number }): ()
        const lines = {};

        for path, time in entries do
            table.insert(lines, `{path},{time}`);
        end;

        writefile(timestamps_path, table.concat(lines, '\n'));
    end;

    function timestamps.set(path: string, time: number): ()
        const entries = timestamps.read();
        entries[path] = time;
        timestamps.write(entries);
    end;

    function timestamps.get(path: string): (number?)
        const entries = timestamps.read();
        return(entries[path]);
    end;

    function timestamps.remove(path: string): ()
        const entries = timestamps.read();
        entries[path] = nil;
        timestamps.write(entries);
    end;
end;

---- classes ----

const handle = {} :: handle; do
    --- methods ---

    function handle.write(this: handle, content: string): (handle)
        -- assertions --

        assert(type(this)    == 'table' and this.write, `expected handle:write to be called as a method`);
        assert(type(content) == 'string',               `handle.write: unexpected argument #2, expect 'content' to be a string, got {type(content)}`);

        -- runtime --

        writefile(this.path, content);
        timestamps.set(this.path, os.time());

        -- exports --

        return(this)
    end

    function handle.append(this: handle, content: string): (handle)
        -- assertions --

        assert(type(this)    == 'table' and this.write, `expected handle:append to be called as a method`);
        assert(type(content) == 'string',               `handle.append: unexpected argument #2, expect 'content' to be a string, got {type(content)}`);

        -- runtime --

        writefile(this.path, `{readfile(this.path) :: string}{content}`);
        timestamps.set(this.path, os.time());

        -- exports --

        return(this)
    end

    function handle.read(this: handle): (string)
        -- assertions --

        assert(type(this) == 'table' and this.write, `expected handle:read to be called as a method`);
        if(not isfile(this.path)) then return('') end;

        -- exports --

        return(readfile(this.path) :: string)
    end

    function handle.size(this: handle): (number)
        -- assertions --

        assert(type(this) == 'table' and this.write, `expected handle:size to be called as a method`);
        if(not isfile(this.path)) then return(0) end;

        -- exports --

        return(#readfile(this.path) :: number)
    end

    function handle.load(this: handle): ((...any) -> (...any))
        -- assertions --

        assert(type(this) == 'table' and this.write, `expected handle:load to be called as a method`);
        if(not isfile(this.path)) then return(function() end) end;

        -- exports --

        return(loadfile(this.path) :: (...any) -> (...any))
    end

    function handle.close(this: handle): ()
        -- assertions --

        assert(type(this) == 'table' and this.write, `expected handle:close to be called as a method`);

        -- exports --

        return table.clear(this);
    end

    function handle.timestamp(this: handle): (number)
        -- assertions --

        assert(type(this) == 'table' and this.write, `expected handle:timestamp to be called as a method`);

        -- exports --

        return(timestamps.get(this.path) :: number);
    end

    --- metatables ---

    handle.__index = handle
end

const fs = {} :: fs; do
    function fs.open(path: string): typeof(setmetatable( { path = string }, typeof(handle)))
        -- assertions --

        assert(type(path) == 'string', `fs.open: unexpected argument #1, expect 'path' to be a string, got {type(path)}`);

        -- exports --

        return(setmetatable(
            {
                path = path
            },

            handle
        ));
    end;

    function fs.read(path: string): (string)
        -- assertions --

        assert(type(path) == 'string', `fs.read: unexpected argument #1, expect 'path' to be a string, got {type(path)}`);

        -- exports --

        return(readfile(path) :: string);
    end;

    function fs.entries(path: string): { string }
        -- assertions --

        assert(type(path) == 'string', `fs.entries: unexpected argument #1, expect 'path' to be a string, got {type(path)}`);

        -- exports --

        return(listfiles(path) :: { string });
    end;

    function fs.file(path: string): (boolean)
        -- assertions --

        assert(type(path) == 'string', `fs.file: unexpected argument #1, expect 'path' to be a string, got {type(path)}`);

        -- exports --

        return(isfile(path));
    end;

    function fs.folder(path: string): (boolean)
        -- assertions --

        assert(type(path) == 'string', `fs.folder: unexpected argument #1, expect 'path' to be a string, got {type(path)}`);

        -- exports --

        return(isfolder(path));
    end;

    function fs.write(path: string, content: string): ()
        -- assertions --

        assert(type(path)    == 'string', `fs.write: unexpected argument #1, expect 'path' to be a string, got {type(path)}`);
        assert(type(content) == 'string', `fs.write: unexpected argument #2, expect 'content' to be a string, got {type(content)}`);

        -- exports --

        writefile(path, content);
        timestamps.set(path, os.time());
    end;

    function fs.load(path: string): ((...any) -> (...any))
        -- assertions --

        assert(type(path) == 'string', `fs.load: unexpected argument #1, expect 'path' to be a string, got {type(path)}`);

        -- exports --

        return(loadfile(path) :: (...any) -> (...any));
    end;

    function fs.make(path: string): ()
        -- assertions --

        assert(type(path) == 'string', `fs.make: unexpected argument #1, expect 'path' to be a string, got {type(path)}`);

        -- exports --

        return(makefolder(path));
    end;

    function fs.remove(path: string): ()
        -- assertions --

        assert(type(path) == 'string', `fs.remove: unexpected argument #1, expect 'path' to be a string, got {type(path)}`);

        -- exports --

        timestamps.remove(path);
        return(delfolder(path));
    end;

    function fs.delete(path: string): ()
        -- assertions --

        assert(type(path) == 'string', `fs.delete: unexpected argument #1, expect 'path' to be a string, got {type(path)}`);

        -- exports --

        timestamps.remove(path);
        return(delfile(path));
    end;

    function fs.timestamp(path: string): (number)
        -- assertions --

        assert(type(path) == 'string', `fs.timestamp: unexpected argument #1, expect 'path' to be a string, got {type(path)}`);

        -- exports --

        return(timestamps.get(path) :: number);
    end;
end;

---- runtime ----

const entries = timestamps.read();
const valid   = {};

for path, time in entries do
    if(isfile(path) or isfolder(path)) then
        valid[path] = time;
    end;
end;

timestamps.write(valid);

---- exports ----

return(fs)
