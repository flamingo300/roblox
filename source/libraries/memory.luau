---- imports ----

local bit64 = require('@types/bit64');

---- functions ----

function memory.readu64(source, offset)
    if('number' == type(source)) then
        local data = memory.readbuffer(source, 8);

        return bit64.buffer(data);
    else
        local data = memory.readbuffer(source, (offset or 0), 8);

        return bit64.buffer(data);
    end
end

---- exports ----

return(memory);
