type uint64__DARKLUA_TYPE_a=vector type uint32__DARKLUA_TYPE_b=number type
uint__DARKLUA_TYPE_c=(uint64__DARKLUA_TYPE_a|uint32__DARKLUA_TYPE_b)type
bit64__DARKLUA_TYPE_d={pair:&((h:number,l:number)->(uint64__DARKLUA_TYPE_a))&((
uint64__DARKLUA_TYPE_a)->(number,number)),buffer:&((b:buffer,o:number?)->(
uint64__DARKLUA_TYPE_a))&((uint64__DARKLUA_TYPE_a,b:buffer?,o:number?)->(buffer?
)),double:&((n:number)->(uint64__DARKLUA_TYPE_a))&((uint64__DARKLUA_TYPE_a)->(
number)),zero:uint64__DARKLUA_TYPE_a,add:(n:uint__DARKLUA_TYPE_c,x:
uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,sub:(n:uint__DARKLUA_TYPE_c,x:
uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,mul:(n:uint__DARKLUA_TYPE_c,x:
uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,div:(n:uint__DARKLUA_TYPE_c,x:
uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,mod:(n:uint__DARKLUA_TYPE_c,x:
uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,pow:(n:uint__DARKLUA_TYPE_c,x:
uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,band:(...uint__DARKLUA_TYPE_c)->
uint64__DARKLUA_TYPE_a,bor:(...uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,
bxor:(...uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,bnot:(n:
uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,lshift:(n:uint__DARKLUA_TYPE_c,d:
uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,rshift:(n:uint__DARKLUA_TYPE_c,d:
uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,lrotate:(n:uint__DARKLUA_TYPE_c,d:
uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,rrotate:(n:uint__DARKLUA_TYPE_c,d:
uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,countlz:(n:uint__DARKLUA_TYPE_c)->
uint64__DARKLUA_TYPE_a,countrz:(n:uint__DARKLUA_TYPE_c)->uint64__DARKLUA_TYPE_a,
eq:(n:uint__DARKLUA_TYPE_c,x:uint__DARKLUA_TYPE_c)->boolean,lt:(n:
uint__DARKLUA_TYPE_c,x:uint__DARKLUA_TYPE_c)->boolean,le:(n:uint__DARKLUA_TYPE_c
,x:uint__DARKLUA_TYPE_c)->boolean,hex:(n:uint__DARKLUA_TYPE_c)->string,decimal:(
n:uint__DARKLUA_TYPE_c)->string}type handle__DARKLUA_TYPE_e={write:(this:
handle__DARKLUA_TYPE_e,content:string)->(handle__DARKLUA_TYPE_e),read:(this:
handle__DARKLUA_TYPE_e)->(string),size:(this:handle__DARKLUA_TYPE_e)->(number),
load:(this:handle__DARKLUA_TYPE_e)->((...any)->(...any)),close:(this:
handle__DARKLUA_TYPE_e)->(),timestamp:(this:handle__DARKLUA_TYPE_e)->(number),
path:string}type fs__DARKLUA_TYPE_f={open:(path:string)->(handle__DARKLUA_TYPE_e
),read:(path:string)->(string),entries:(path:string)->{string},file:(path:string
)->(boolean),folder:(path:string)->(boolean),write:(path:string,content:string
)->(),load:(path:string)->((...any)->(...any)),make:(path:string)->(),remove:(
path:string)->(),delete:(path:string)->(),timestamp:(path:string)->(number)}
local __DARKLUA_BUNDLE_MODULES={cache={}::any}do do local function __modImpl()
local bit32,buffer,vector,string=bit32,buffer,vector,string local band,bor,bxor,
bnot,lshift,rshift,countlz,countrz=bit32.band,bit32.bor,bit32.bxor,bit32.bnot,
bit32.lshift,bit32.rshift,bit32.countlz,bit32.countrz local format=string.format
local char=string.char local v=vector.create local m32,m22,m20,ll,lm=4294967296,
0x3fffff,0xfffff,4194304,1024 local function normalize(uint64:
uint64__DARKLUA_TYPE_a):vector return v(band(uint64.x,m22),band(uint64.y,m20),
band(uint64.z,m22))end local bit64={}::bit64__DARKLUA_TYPE_d do local function 
uint64(n:uint__DARKLUA_TYPE_c):uint64__DARKLUA_TYPE_a return if('number'==type(n
))then bit64.double(n)else n::uint64__DARKLUA_TYPE_a end local function shift(d:
uint__DARKLUA_TYPE_c):number if('number'==type(d))then return d end local h,l=
bit64.pair(d::uint64__DARKLUA_TYPE_a)return if(h~=0)then 64 else l end function
bit64.pair(h,l)if('vector'==type(h))then local x:number,y:number,z:number=h.x,h.
y,h.z return x*lm+y//lm,(y%lm)*ll+z end assert('number'==type(h)and'number'==
type(l),'bit64.pair: expected (number, number) or uint64')return v(h//lm,(h%lm)*
lm+l//ll,l%ll)end function bit64.buffer(b,o,n)if('vector'==type(b))then local h,
l=bit64.pair(b)local block=if(not o)then buffer.create(8)else o n=n or 0 buffer.
writeu32(block,n,h)buffer.writeu32(block,n+4,l)return(block)end if('number'==
type(b))then local h,l=0,b local block=if(not o)then buffer.create(8)else o n=n
or 0 buffer.writeu32(block,n,h)buffer.writeu32(block,n+4,l)return(block)end
assert('buffer'==type(b),'bit64.buffer: expected buffer or uint64')o=o or 0
return bit64.pair(buffer.readu32(b::buffer,o::number),buffer.readu32(b::buffer,(
o+4)::number))end function bit64.double(n)if('vector'==type(n))then local h,l=
bit64.pair(n)return h*m32+l end assert('number'==type(n),
'bit64.double: expected (number) or uint64')return bit64.pair(n//m32,n%m32)end
bit64.zero=vector.zero function bit64.add(n,x)n=type(n)=='number'and bit64.
double(n)or n x=type(x)=='number'and bit64.double(x)or x local prime=n+x prime+=
v(0,(if prime.z>m22 then 1 else 0),0)prime+=v((if prime.y>m20 then 1 else 0),0,0
)return normalize(prime)end function bit64.sub(n,x)n=type(n)=='number'and bit64.
double(n)or n x=type(x)=='number'and bit64.double(x)or x local prime=n-x prime-=
v(0,(if prime.z<0 then 1 else 0),0)prime-=v((if prime.y<0 then 1 else 0),0,0)
return normalize(prime)end function bit64.mul(n,x)n=type(n)=='number'and bit64.
double(n)or n x=type(x)=='number'and bit64.double(x)or x local prime=n*x prime+=
v(0,(if prime.z>m22 then 1 else 0),0)prime+=v((if prime.y>m20 then 1 else 0),0,0
)return normalize(prime)end function bit64.div(n,x)n=type(n)=='number'and bit64.
double(n)or n x=type(x)=='number'and bit64.double(x)or x return normalize(n//x)
end function bit64.mod(n,x)n=type(n)=='vector'and bit64.double(n)or n x=type(x)
=='vector'and bit64.double(x)or x return bit64.double(n%x)end function bit64.pow
(n,x)n=uint64(n)x=uint64(x)if(x.x==0 and x.y==0 and x.z==0)then return bit64.
pair(0,1)end if(x.x==0 and x.y==0 and x.z==1)then return n end local result=
bit64.pair(0,1)while(true)do local _,xl=bit64.pair(x)if(band(xl,1)~=0)then
result=bit64.mul(result,n)end x=bit64.rshift(x,1)if(x.x==0 and x.y==0 and x.z==0
)then break end n=bit64.mul(n,n)end return result end function bit64.band(...)
local args={...}local r=uint64(args[1])for i=2,#args do local x=uint64(args[i])r
=v(band(r.x,x.x),band(r.y,x.y),band(r.z,x.z))end return r end function bit64.bor
(...)local args={...}local r=uint64(args[1])for i=2,#args do local x=uint64(args
[i])r=v(bor(r.x,x.x),bor(r.y,x.y),bor(r.z,x.z))end return r end function bit64.
bxor(...)local args={...}local r=uint64(args[1])for i=2,#args do local x=uint64(
args[i])r=v(bxor(r.x,x.x),bxor(r.y,x.y),bxor(r.z,x.z))end return r end function
bit64.bnot(n)n=uint64(n)return v(band(bnot(n.x),m22),band(bnot(n.y),m20),band(
bnot(n.z),m22))end function bit64.lshift(n,d)n=uint64(n)local s=shift(d)if(s<=0)
then return normalize(n)end if(s>=64)then return bit64.zero end local h,l=bit64.
pair(n)if(s>=32)then return bit64.pair(lshift(l,s-32),0)end return bit64.pair(
bor(lshift(h,s),rshift(l,32-s)),lshift(l,s))end function bit64.rshift(n,d)n=
uint64(n)local s=shift(d)if(s<=0)then return normalize(n)end if(s>=64)then
return bit64.zero end local h,l=bit64.pair(n)if(s>=32)then return bit64.pair(0,
rshift(h,s-32))end return bit64.pair(rshift(h,s),bor(rshift(l,s),lshift(h,32-s))
)end function bit64.lrotate(n,d)n=uint64(n)local s=shift(d)%64 if(s==0)then
return normalize(n)end return bit64.bor(bit64.lshift(n,s),bit64.rshift(n,64-s))
end function bit64.rrotate(n,d)n=uint64(n)local s=shift(d)%64 if(s==0)then
return normalize(n)end return bit64.bor(bit64.rshift(n,s),bit64.lshift(n,64-s))
end function bit64.countlz(n)n=uint64(n)local h,l=bit64.pair(n)if(h==0)then
return 32+countlz(l)end return countlz(h)end function bit64.countrz(n)n=uint64(n
)local h,l=bit64.pair(n)if(l==0)then return 32+countrz(h)end return countrz(l)
end function bit64.eq(n,x)n,x=uint64(n),uint64(x)return n.x==x.x and n.y==x.y
and n.z==x.z end function bit64.lt(n,x)n,x=uint64(n),uint64(x)local nh,nl=bit64.
pair(n)local xh,xl=bit64.pair(x)if(nh~=xh)then return nh<xh end return nl<xl end
function bit64.le(n,x)return bit64.lt(n,x)or bit64.eq(n,x)end function bit64.hex
(n)return format('0x%08X%08X',bit64.pair(uint64(n)))end function bit64.decimal(n
)local h,l=bit64.pair(uint64(n))if(h==0 and l==0)then return'0'end local o,d,c=
'',{},0 while(h~=0 or l~=0)do local q=h//10 local r=h%10 local t=r*m32+l h,l=q,t
//10 c+=1 d[c]=t%10 end for i=c,1,-1 do o..=char(48+d[i])end return o end end
return(bit64)end function __DARKLUA_BUNDLE_MODULES.a():typeof(__modImpl())local
v=__DARKLUA_BUNDLE_MODULES.cache.a if not v then v={c=__modImpl()}
__DARKLUA_BUNDLE_MODULES.cache.a=v end return v.c end end do local function 
__modImpl()local timestamps_path='.timestamps.txt'local timestamps={}::{[string]
:number}do function timestamps.read():{[string]:number}if(not isfile(
timestamps_path))then writefile(timestamps_path,'')return{}end local entries={}
::{[string]:number}local content=readfile(timestamps_path)::string for line in
content:gmatch('[^\r\n]+')do local path,time=line:match('^([^,]+),(%d+)$')if(
path and time)then entries[path]=tonumber(time)::number end end return(entries)
end function timestamps.write(entries:{[string]:number}):()local lines={}for
path,time in entries do table.insert(lines,`{path},{time}`)end writefile(
timestamps_path,table.concat(lines,'\n'))end function timestamps.set(path:string
,time:number):()local entries=timestamps.read()entries[path]=time timestamps.
write(entries)end function timestamps.get(path:string):(number?)local entries=
timestamps.read()return(entries[path])end function timestamps.remove(path:string
):()local entries=timestamps.read()entries[path]=nil timestamps.write(entries)
end end local handle={}::handle__DARKLUA_TYPE_e do function handle.write(this:
handle__DARKLUA_TYPE_e,content:string):(handle__DARKLUA_TYPE_e)assert(type(this)
=='table'and this.write,`expected handle:write to be called as a method`)assert(
type(content)=='string',`handle.write: unexpected argument #2, expect 'content' to be a string, got {
type(content)}`)writefile(this.path,content)timestamps.set(this.path,os.time())
return(this)end function handle.append(this:handle__DARKLUA_TYPE_e,content:
string):(handle__DARKLUA_TYPE_e)assert(type(this)=='table'and this.write,`expected handle:append to be called as a method`
)assert(type(content)=='string',`handle.append: unexpected argument #2, expect 'content' to be a string, got {
type(content)}`)writefile(this.path,`{readfile(this.path)::string}{content}`)
timestamps.set(this.path,os.time())return(this)end function handle.read(this:
handle__DARKLUA_TYPE_e):(string)assert(type(this)=='table'and this.write,`expected handle:read to be called as a method`
)if(not isfile(this.path))then return('')end return(readfile(this.path)::string)
end function handle.size(this:handle__DARKLUA_TYPE_e):(number)assert(type(this)
=='table'and this.write,`expected handle:size to be called as a method`)if(not
isfile(this.path))then return(0)end return(#readfile(this.path)::number)end
function handle.load(this:handle__DARKLUA_TYPE_e):((...any)->(...any))assert(
type(this)=='table'and this.write,`expected handle:load to be called as a method`
)if(not isfile(this.path))then return(function()end)end return(loadfile(this.
path)::(...any)->(...any))end function handle.close(this:handle__DARKLUA_TYPE_e)
:()assert(type(this)=='table'and this.write,`expected handle:close to be called as a method`
)return table.clear(this)end function handle.timestamp(this:
handle__DARKLUA_TYPE_e):(number)assert(type(this)=='table'and this.write,`expected handle:timestamp to be called as a method`
)return(timestamps.get(this.path)::number)end handle.__index=handle end local fs
={}::fs__DARKLUA_TYPE_f do function fs.open(path:string):typeof(setmetatable({
path=string},typeof(handle)))assert(type(path)=='string',`fs.open: unexpected argument #1, expect 'path' to be a string, got {
type(path)}`)return(setmetatable({path=path},handle))end function fs.read(path:
string):(string)assert(type(path)=='string',`fs.read: unexpected argument #1, expect 'path' to be a string, got {
type(path)}`)return(readfile(path)::string)end function fs.entries(path:string):
{string}assert(type(path)=='string',`fs.entries: unexpected argument #1, expect 'path' to be a string, got {
type(path)}`)return(listfiles(path)::{string})end function fs.file(path:string):
(boolean)assert(type(path)=='string',`fs.file: unexpected argument #1, expect 'path' to be a string, got {
type(path)}`)return(isfile(path))end function fs.folder(path:string):(boolean)
assert(type(path)=='string',`fs.folder: unexpected argument #1, expect 'path' to be a string, got {
type(path)}`)return(isfolder(path))end function fs.write(path:string,content:
string):()assert(type(path)=='string',`fs.write: unexpected argument #1, expect 'path' to be a string, got {
type(path)}`)assert(type(content)=='string',`fs.write: unexpected argument #2, expect 'content' to be a string, got {
type(content)}`)writefile(path,content)timestamps.set(path,os.time())end
function fs.load(path:string):((...any)->(...any))assert(type(path)=='string',`fs.load: unexpected argument #1, expect 'path' to be a string, got {
type(path)}`)return(loadfile(path)::(...any)->(...any))end function fs.make(path
:string):()assert(type(path)=='string',`fs.make: unexpected argument #1, expect 'path' to be a string, got {
type(path)}`)return(makefolder(path))end function fs.remove(path:string):()
assert(type(path)=='string',`fs.remove: unexpected argument #1, expect 'path' to be a string, got {
type(path)}`)timestamps.remove(path)return(delfolder(path))end function fs.
delete(path:string):()assert(type(path)=='string',`fs.delete: unexpected argument #1, expect 'path' to be a string, got {
type(path)}`)timestamps.remove(path)return(delfile(path))end function fs.
timestamp(path:string):(number)assert(type(path)=='string',`fs.timestamp: unexpected argument #1, expect 'path' to be a string, got {
type(path)}`)return(timestamps.get(path)::number)end end local entries=
timestamps.read()local valid={}for path,time in entries do if(isfile(path)or
isfolder(path))then valid[path]=time end end timestamps.write(valid)return(fs)
end function __DARKLUA_BUNDLE_MODULES.b():typeof(__modImpl())local v=
__DARKLUA_BUNDLE_MODULES.cache.b if not v then v={c=__modImpl()}
__DARKLUA_BUNDLE_MODULES.cache.b=v end return v.c end end do local function 
__modImpl()return{get=function(args:{url:string,content:string}):string return
game:HttpGet(args.url,args.content)end,post=function(args:{url:string,content:
string,type:string,accept:string,cookie:string,referrer:string,origin:string}):
string return game:HttpPost(args.url,args.content,args.type,args.accept,args.
cookie,args.referrer,args.origin)end}end function __DARKLUA_BUNDLE_MODULES.c():
typeof(__modImpl())local v=__DARKLUA_BUNDLE_MODULES.cache.c if not v then v={c=
__modImpl()}__DARKLUA_BUNDLE_MODULES.cache.c=v end return v.c end end end _G.
bit64=__DARKLUA_BUNDLE_MODULES.a()_G.fs=__DARKLUA_BUNDLE_MODULES.b()_G.http=
__DARKLUA_BUNDLE_MODULES.c()