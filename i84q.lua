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
path:string)->(),delete:(path:string)->(),timestamp:(path:string)->(number)}type
InternalIdentity__DARKLUA_TYPE_g<T...> =setmetatable<{_head:number,
_collect_queue:{{[number]:unknown,seen_by:number}},_collector_count:number,
_collector_mask:number,[number]:(T...)->()},typeof(signal)>type
Identity__DARKLUA_TYPE_h<T...> ={fire:(self:Identity__DARKLUA_TYPE_h<T...>,T...
)->(),connect:(self:Identity__DARKLUA_TYPE_h<T...>,callback:(T...)->())->()->(),
once:(self:Identity__DARKLUA_TYPE_h<T...>,callback:(T...)->())->()->(),wait:(
self:Identity__DARKLUA_TYPE_h<T...>)->T...,disconnect_all:(self:
Identity__DARKLUA_TYPE_h<T...>)->(),delete:(self:Identity__DARKLUA_TYPE_h<T...>
)->(),collect:(self:Identity__DARKLUA_TYPE_h<T...>)->()->T...}type
UDim__DARKLUA_TYPE_i={Scale:number,Offset:number}type UDim2__DARKLUA_TYPE_j={X:
UDim__DARKLUA_TYPE_i,Y:UDim__DARKLUA_TYPE_i}type Point3D__DARKLUA_TYPE_k={
Position:vector,Active:boolean,Destroy:(self:Point3D__DARKLUA_TYPE_k)->()}type
Point2D__DARKLUA_TYPE_l={Position:vector,Active:boolean,Destroy:(self:
Point2D__DARKLUA_TYPE_l)->()}type PointInstance__DARKLUA_TYPE_m={Instance:
Instance?,Active:boolean,Destroy:(self:PointInstance__DARKLUA_TYPE_m)->(),CFrame
:CFrame?,Size:vector?}type PointModel__DARKLUA_TYPE_n={Instance:Instance?,Follow
:Instance?,Active:boolean,Destroy:(self:PointModel__DARKLUA_TYPE_n)->(),CFrame:
CFrame?,Size:vector?,_RelPos:vector?,_RelR:vector?,_RelU:vector?,_RelL:vector?,
_Size:vector?}type UDim__DARKLUA_TYPE_o={Scale:number,Offset:number}type
UDim2__DARKLUA_TYPE_p={X:UDim__DARKLUA_TYPE_o,Y:UDim__DARKLUA_TYPE_o}type
Point3D__DARKLUA_TYPE_q={Position:vector,Active:boolean,Destroy:(self:
Point3D__DARKLUA_TYPE_q)->()}type Point2D__DARKLUA_TYPE_r={Position:vector,
Active:boolean,Destroy:(self:Point2D__DARKLUA_TYPE_r)->()}type
PointInstance__DARKLUA_TYPE_s={Instance:Instance?,Active:boolean,CFrame:CFrame?,
Size:vector?}type PointModel__DARKLUA_TYPE_t={Instance:Instance?,Follow:Instance
?,Active:boolean,CFrame:CFrame?,Size:vector?}type Point__DARKLUA_TYPE_u=
Point3D__DARKLUA_TYPE_q|Point2D__DARKLUA_TYPE_r|PointInstance__DARKLUA_TYPE_s|
PointModel__DARKLUA_TYPE_t|Instance type Attachment__DARKLUA_TYPE_v={Link:
Point__DARKLUA_TYPE_u?,From:Point__DARKLUA_TYPE_u?,To:Point__DARKLUA_TYPE_u?,
Size:UDim2__DARKLUA_TYPE_p,Position:UDim2__DARKLUA_TYPE_p,AnchorPoint:vector}
type Cluster__DARKLUA_TYPE_w={Attachments:{[any]:Attachment__DARKLUA_TYPE_v},
Active:boolean,Paused:boolean,Connection:any,Pause:(self:Cluster__DARKLUA_TYPE_w
)->(),Resume:(self:Cluster__DARKLUA_TYPE_w)->(),Destroy:(self:
Cluster__DARKLUA_TYPE_w)->()}type Drawing_module__DARKLUA_TYPE_x={attach:(
descriptor:{[any]:{Link:Point__DARKLUA_TYPE_u?,From:Point__DARKLUA_TYPE_u?,To:
Point__DARKLUA_TYPE_u?,Size:UDim2__DARKLUA_TYPE_p?,Position:
UDim2__DARKLUA_TYPE_p?,AnchorPoint:vector?}})->Cluster__DARKLUA_TYPE_w}local
__DARKLUA_BUNDLE_MODULES={cache={}::any}do do local function __modImpl()local
bit32,buffer,vector,string=bit32,buffer,vector,string local band,bor,bxor,bnot,
lshift,rshift,countlz,countrz=bit32.band,bit32.bor,bit32.bxor,bit32.bnot,bit32.
lshift,bit32.rshift,bit32.countlz,bit32.countrz local format=string.format local
char=string.char local v=vector.create local m32,m22,m20,ll,lm=4294967296,
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
__modImpl()return{get=function(args):string return game:HttpGet(args.url,args.
content)end,post=function(args):string return game:HttpPost(args.url,args.
content,args.type,args.accept,args.cookie,args.referrer,args.origin)end}::
typeof(http)end function __DARKLUA_BUNDLE_MODULES.c():typeof(__modImpl())local v
=__DARKLUA_BUNDLE_MODULES.cache.c if not v then v={c=__modImpl()}
__DARKLUA_BUNDLE_MODULES.cache.c=v end return v.c end end do local function 
__modImpl()local task=task local tspawn=task.spawn local NIL_TABLE={}local
free_thread:thread?=nil local function bitmask(idx:number)return 2^idx end local 
function set_bit(int:number,mask:number):number return bit32.bor(int,mask)end
local function get_bit(int:number,mask:number):boolean return bit32.band(int,
mask)~=0 end local function all_bits_set(bit_stack:number,mask:number):boolean
return bit32.band(bit_stack,mask)==mask end local function deleted_signal_err()
error('Cannot use a deleted signal',2)end local error_tbl={fire=
deleted_signal_err,connect=deleted_signal_err,once=deleted_signal_err,wait=
deleted_signal_err,disconnect_all=deleted_signal_err}local function yield_loop()
while true do local sig:InternalIdentity__DARKLUA_TYPE_g<any>,arg1,arg2,arg3,
arg4,arg5,arg6,arg7,arg8,arg9,arg10=coroutine.yield()local ref=free_thread
free_thread=nil while sig._head~=0 do sig._head-=1 sig[sig._head+1](arg1,arg2,
arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10)end free_thread=ref end end local
signal={}signal.__index=signal local function constructor<T...>():
InternalIdentity__DARKLUA_TYPE_g<T...>return setmetatable({_head=0,
_collect_queue={}::any,_collector_count=0,_collector_mask=0},signal)end function
signal.connect<T...>(self:InternalIdentity__DARKLUA_TYPE_g<T...>,callback:(T...
)->())table.insert(self,callback)local function disconnecter()local index=table.
find(self,callback)if index then table.remove(self,index)end end return
disconnecter end function signal.fire<T...>(self:
InternalIdentity__DARKLUA_TYPE_g<T...>,...)if self._collector_count>=1 then
table.insert(self._collect_queue,{...,seen_by=0})end self._head=#self while self
._head~=0 do if not free_thread then free_thread=tspawn(yield_loop)end(tspawn::
any)(free_thread,self,...)end end function signal.once<T...>(self:
InternalIdentity__DARKLUA_TYPE_g<T...>,callback:(T...)->())local disconnect
disconnect=self:connect(function(...)assert(disconnect~=nil,'Luau')disconnect()
callback(...)end)end function signal.wait<T...>(self:
InternalIdentity__DARKLUA_TYPE_g<T...>):T...local running=coroutine.running()
self:once(function(...)assert(coroutine.status(running)=='suspended',
[[:wait() called, then another thread resumed the waiting thread. Please dont do that :(]]
)tspawn(running,...)end)return coroutine.yield()end function signal.collect<T...
>(self:InternalIdentity__DARKLUA_TYPE_g<T...>):()->T...self._collector_count+=1
local mask=bitmask(self._collector_count-1)self._collector_mask=bit32.lshift(1,
self._collector_count-1)local function iter():...unknown local collect_queue=
self._collect_queue for i=#collect_queue,1,-1 do local item=collect_queue[i]if
get_bit(item.seen_by,mask)then continue end local next_seen_by=set_bit(item.
seen_by,mask)if all_bits_set(next_seen_by,self._collector_mask)then table.
remove(collect_queue,i)return table.unpack(item)end item.seen_by=next_seen_by
return table.unpack(item)end return end return iter::any end function signal.
disconnect_all<T...>(self:InternalIdentity__DARKLUA_TYPE_g<T...>)table.move(
NIL_TABLE,1,#self,1,self)end function signal.delete<T...>(self:
InternalIdentity__DARKLUA_TYPE_g<T...>):()self:disconnect_all()setmetatable(self
,error_tbl)end return constructor::<T...>()->Identity__DARKLUA_TYPE_h<T...>end
function __DARKLUA_BUNDLE_MODULES.d():typeof(__modImpl())local v=
__DARKLUA_BUNDLE_MODULES.cache.d if not v then v={c=__modImpl()}
__DARKLUA_BUNDLE_MODULES.cache.d=v end return v.c end end do local function 
__modImpl()local UDim2={}do function UDim2.new(x0:number,x1:number,y0:number,y1:
number):UDim2__DARKLUA_TYPE_j assert(type(x0)=='number',`invalid argument #1 to 'UDim2.new': expected number, got {
type(x0)}`)assert(type(x1)=='number',`invalid argument #2 to 'UDim2.new': expected number, got {
type(x1)}`)assert(type(y0)=='number',`invalid argument #3 to 'UDim2.new': expected number, got {
type(y0)}`)assert(type(y1)=='number',`invalid argument #4 to 'UDim2.new': expected number, got {
type(y1)}`)return setmetatable({X={Scale=x0,Offset=x1},Y={Scale=y0,Offset=y1}},
UDim2)::any end function UDim2.fromScale(x:number,y:number):
UDim2__DARKLUA_TYPE_j assert(type(x)=='number',`invalid argument #1 to 'UDim2.fromScale': expected number, got {
type(x)}`)assert(type(y)=='number',`invalid argument #2 to 'UDim2.fromScale': expected number, got {
type(y)}`)return UDim2.new(x,0,y,0)end function UDim2.fromOffset(x:number,y:
number):UDim2__DARKLUA_TYPE_j assert(type(x)=='number',`invalid argument #1 to 'UDim2.fromOffset': expected number, got {
type(x)}`)assert(type(y)=='number',`invalid argument #2 to 'UDim2.fromOffset': expected number, got {
type(y)}`)return UDim2.new(0,x,0,y)end UDim2.__index=UDim2 end return UDim2 end
function __DARKLUA_BUNDLE_MODULES.e():typeof(__modImpl())local v=
__DARKLUA_BUNDLE_MODULES.cache.e if not v then v={c=__modImpl()}
__DARKLUA_BUNDLE_MODULES.cache.e=v end return v.c end end do local function 
__modImpl()local Point3D={}do function Point3D.new(position:vector):
Point3D__DARKLUA_TYPE_k assert(type(position)=='vector',`invalid argument #1 to 'Point3D.new': expected vector, got {
type(position)}`)return setmetatable({Position=position,Active=true},Point3D)::
any end function Point3D.Destroy(self:Point3D__DARKLUA_TYPE_k):()self.Active=
false(self::any).Position=nil end Point3D.__index=Point3D end return Point3D end
function __DARKLUA_BUNDLE_MODULES.f():typeof(__modImpl())local v=
__DARKLUA_BUNDLE_MODULES.cache.f if not v then v={c=__modImpl()}
__DARKLUA_BUNDLE_MODULES.cache.f=v end return v.c end end do local function 
__modImpl()local Point2D={}do function Point2D.new(position:vector):
Point2D__DARKLUA_TYPE_l assert(type(position)=='vector',`invalid argument #1 to 'Point2D.new': expected vector, got {
type(position)}`)return setmetatable({Position=position,Active=true},Point2D)::
any end function Point2D.Destroy(self:Point2D__DARKLUA_TYPE_l):()self.Active=
false(self::any).Position=nil end Point2D.__index=Point2D end return Point2D end
function __DARKLUA_BUNDLE_MODULES.g():typeof(__modImpl())local v=
__DARKLUA_BUNDLE_MODULES.cache.g if not v then v={c=__modImpl()}
__DARKLUA_BUNDLE_MODULES.cache.g=v end return v.c end end do local function 
__modImpl()local PointInstance={}do function PointInstance.new(instance:Instance
):PointInstance__DARKLUA_TYPE_m assert(typeof(instance)=='Instance',`invalid argument #1 to 'PointInstance.new': expected Instance, got {
typeof(instance)}`)return setmetatable({Instance=instance,Active=true},
PointInstance)::any end function PointInstance.Destroy(self:
PointInstance__DARKLUA_TYPE_m):()self.Active=false(self::any).Instance=nil end
function PointInstance.__index(self:PointInstance__DARKLUA_TYPE_m,key:string):
any local inst=rawget(self,'Instance'::any)if(key=='CFrame')then return if(inst)
then(inst::any).CFrame else nil elseif(key=='Size')then return if(inst)then(inst
::any).Size else nil end return(PointInstance::any)[key]end end return
PointInstance end function __DARKLUA_BUNDLE_MODULES.h():typeof(__modImpl())local
v=__DARKLUA_BUNDLE_MODULES.cache.h if not v then v={c=__modImpl()}
__DARKLUA_BUNDLE_MODULES.cache.h=v end return v.c end end do local function 
__modImpl()local huge=math.huge local max=math.max local min=math.min local
create=vector.create local dot=vector.dot local CornerSigns=table.freeze({table.
freeze({-1,-1,-1}),table.freeze({1,-1,-1}),table.freeze({-1,1,-1}),table.freeze(
{1,1,-1}),table.freeze({-1,-1,1}),table.freeze({1,-1,1}),table.freeze({-1,1,1}),
table.freeze({1,1,1})}::any)local PointModel={}do function PointModel.new(
instance:Instance):PointModel__DARKLUA_TYPE_n assert(typeof(instance)==
'Instance',`invalid argument #1 to 'PointModel.new': expected Instance, got {
typeof(instance)}`)assert(instance:IsA('Model')or instance:IsA('Folder'),`invalid argument #1 to 'PointModel.new': expected Model or Folder, got {
instance.ClassName}`)local follow:Instance?,points:{vector},n:number=nil,{},0
for _,d in instance:GetDescendants()do if(d:IsA('BasePart'))then follow=follow
or d local cframe:CFrame=(d::any).CFrame local size:vector=(d::any).Size local
half:vector=size*0.5 local position,right_vector,up_vector,look_vector=cframe.
Position,cframe.RightVector,cframe.UpVector,cframe.LookVector local hx:number,hy
:number,hz:number=half.x,half.y,half.z for _,s in CornerSigns do local lx:number
,ly:number,lz:number=(s::any)[1]*hx,(s::any)[2]*hy,(s::any)[3]*hz n+=1 points[n]
=position+right_vector*lx+up_vector*ly+look_vector*lz end end end if(instance:
IsA('Model'))then follow=if(follow)then instance.PrimaryPart else follow end if(
n==0)then return setmetatable({Instance=instance,Active=true,Follow=follow,
_RelPos=nil,_RelR=nil,_RelU=nil,_RelL=nil,_Size=vector.zero},PointModel)::any
end local sx:number=0 local sy:number=0 local sz:number=0 for i=1,n do local p:
vector=points[i]sx+=p.x sy+=p.y sz+=p.z end local inv_n:number=1/n local mx:
number=sx*inv_n local my:number=sy*inv_n local mz:number=sz*inv_n local cxx:
number=0 local cxy:number=0 local cxz:number=0 local cyy:number=0 local cyz:
number=0 local czz:number=0 for i=1,n do local p:vector=points[i]local x:number=
p.x-mx local y:number=p.y-my local z:number=p.z-mz cxx+=x*x cxy+=x*y cxz+=x*z
cyy+=y*y cyz+=y*z czz+=z*z end cxx*=inv_n cxy*=inv_n cxz*=inv_n cyy*=inv_n cyz*=
inv_n czz*=inv_n local function normalize(x:number,y:number,z:number):(number,
number,number)local m:number=(x*x+y*y+z*z)^0.5 if(m==0)then return 1,0,0 end
local inv:number=1/m return x*inv,y*inv,z*inv end local vx:number=1 local vy:
number=0 local vz:number=0 for _=1,8 do local nx:number=cxx*vx+cxy*vy+cxz*vz
local ny:number=cxy*vx+cyy*vy+cyz*vz local nz:number=cxz*vx+cyz*vy+czz*vz vx,vy,
vz=normalize(nx,ny,nz)end local ax:number=0 local ay:number=1 local az:number=0
local dot:number=vx*ax+vy*ay+vz*az if((if(dot<0)then-dot else dot)>0.9)then ax,
ay,az=0,0,1 end local ux:number,uy:number,uz:number=normalize(vy*az-vz*ay,vz*ax-
vx*az,vx*ay-vy*ax)local wx:number,wy:number,wz:number=normalize(vy*uz-vz*uy,vz*
ux-vx*uz,vx*uy-vy*ux)local min1:number=huge local min2:number=huge local min3:
number=huge local max1:number=-huge local max2:number=-huge local max3:number=-
huge for i=1,n do local p:vector=points[i]local x:number=p.x-mx local y:number=p
.y-my local z:number=p.z-mz local d1:number=x*vx+y*vy+z*vz local d2:number=x*ux+
y*uy+z*uz local d3:number=x*wx+y*wy+z*wz min1=min(min1,d1)max1=max(max1,d1)min2=
min(min2,d2)max2=max(max2,d2)min3=min(min3,d3)max3=max(max3,d3)end local cx:
number=(min1+max1)*0.5 local cy:number=(min2+max2)*0.5 local cz:number=(min3+
max3)*0.5 local size:vector=create(max1-min1,max2-min2,max3-min3)local
world_center:vector=create(mx+vx*cx+ux*cy+wx*cz,my+vy*cx+uy*cy+wy*cz,mz+vz*cx+uz
*cy+wz*cz)local rel_pos:vector=world_center local rel_r:vector=create(vx,vy,vz)
local rel_u:vector=create(ux,uy,uz)local rel_l:vector=create(wx,wy,wz)if(follow)
then local fcf:CFrame=(follow::any).CFrame local fpos:vector=fcf.Position local
fr:vector=fcf.RightVector local fu:vector=fcf.UpVector local fl:vector=fcf.
LookVector local dx:vector=world_center-fpos rel_pos=create(dot(dx,fr),dot(dx,fu
),dot(dx,fl))local rv:vector=create(vx,vy,vz)local uv:vector=create(ux,uy,uz)
local lv:vector=create(wx,wy,wz)rel_r=create(dot(rv,fr),dot(rv,fu),dot(rv,fl))
rel_u=create(dot(uv,fr),dot(uv,fu),dot(uv,fl))rel_l=create(dot(lv,fr),dot(lv,fu)
,dot(lv,fl))end return setmetatable({Instance=instance,Active=true,Follow=follow
,_RelPos=rel_pos,_RelR=rel_r,_RelU=rel_u,_RelL=rel_l,_Size=size},PointModel)::
any end function PointModel.Destroy(self:PointModel__DARKLUA_TYPE_n):()self.
Active=false(self::any).Instance=nil(self::any).Follow=nil(self::any)._RelPos=
nil(self::any)._RelR=nil(self::any)._RelU=nil(self::any)._RelL=nil(self::any).
_Size=nil end function PointModel.__index(self:PointModel__DARKLUA_TYPE_n,key:
string):any local inst=rawget(self,'Instance'::any)if(key=='CFrame')then if(not
inst)then return nil end local follow=rawget(self,'Follow'::any)local rel_pos=
rawget(self,'_RelPos'::any)local rel_r=rawget(self,'_RelR'::any)local rel_u=
rawget(self,'_RelU'::any)local rel_l=rawget(self,'_RelL'::any)if(not(rel_pos and
rel_r and rel_u and rel_l))then return nil end if(follow)then local fcf:CFrame=(
follow::any).CFrame local fpos:vector=fcf.Position local fr:vector=fcf.
RightVector local fu:vector=fcf.UpVector local fl:vector=fcf.LookVector local
w_center:vector=fpos+fr*(rel_pos::vector).x+fu*(rel_pos::vector).y+fl*(rel_pos::
vector).z local w_r:vector=fr*(rel_r::vector).x+fu*(rel_r::vector).y+fl*(rel_r::
vector).z local w_u:vector=fr*(rel_u::vector).x+fu*(rel_u::vector).y+fl*(rel_u::
vector).z local w_l:vector=fr*(rel_l::vector).x+fu*(rel_l::vector).y+fl*(rel_l::
vector).z return CFrame.fromMatrix(w_center,w_r,w_u,w_l)end return CFrame.
fromMatrix(rel_pos::vector,rel_r::vector,rel_u::vector,rel_l::vector)elseif(key
=='Size')then if(not inst)then return nil end return rawget(self,'_Size'::any)
end return(PointModel::any)[key]end end return PointModel end function
__DARKLUA_BUNDLE_MODULES.i():typeof(__modImpl())local v=__DARKLUA_BUNDLE_MODULES
.cache.i if not v then v={c=__modImpl()}__DARKLUA_BUNDLE_MODULES.cache.i=v end
return v.c end end do local function __modImpl()local vector=vector local table=
table local math=math local game=game local create=vector.create local ceil=
vector.ceil local zero=vector.zero local huge=math.huge local max=math.max local
min=math.min local UDim2=__DARKLUA_BUNDLE_MODULES.e()local PointModel=
__DARKLUA_BUNDLE_MODULES.i()local Point2D_mod=__DARKLUA_BUNDLE_MODULES.g()local
RunService=game:GetService('RunService')local Weak=table.freeze({__mode='k'}::
any)local CornerSigns=table.freeze({table.freeze({-1,-1,-1}::any),table.freeze({
1,-1,-1}::any),table.freeze({-1,1,-1}::any),table.freeze({1,1,-1}::any),table.
freeze({-1,-1,1}::any),table.freeze({1,-1,1}::any),table.freeze({-1,1,1}::any),
table.freeze({1,1,1}::any)}::any)local ModelPointCache=setmetatable({},Weak)
local DefaultSize=UDim2.fromScale(1,1)local DefaultPosition=UDim2.fromScale(0,0)
local DefaultAnchor=create(0,0)local Prototype={}::Cluster__DARKLUA_TYPE_w do
Prototype.__index=Prototype function Prototype.Pause(self:
Cluster__DARKLUA_TYPE_w):()self.Paused=true end function Prototype.Resume(self:
Cluster__DARKLUA_TYPE_w):()self.Paused=false end function Prototype.Destroy(self
:Cluster__DARKLUA_TYPE_w):()self.Active=false local conn=rawget(self,
'Connection'::any)if(conn)then(conn::any):Disconnect()rawset(self,'Connection'::
any,nil)end local atts=rawget(self,'Attachments'::any)if(atts)then for draw_obj,
_ in atts do(draw_obj::any):Remove()end table.clear(atts::any)end end end@native
local function project(cf:CFrame,size:vector,camera:any):vector local half:
vector=size*0.5 local pos:vector=cf.Position local r:vector=cf.RightVector local
u:vector=cf.UpVector local l:vector=cf.LookVector local hx:number=half.x local
hy:number=half.y local hz:number=half.z local min_x:number=huge local min_y:
number=huge local max_x:number=-huge local max_y:number=-huge local projected:
boolean=false for _,s in CornerSigns do local sx:number=(s::any)[1]local sy:
number=(s::any)[2]local sz:number=(s::any)[3]local lx:number=sx*hx local ly:
number=sy*hy local lz:number=sz*hz local w:vector=pos+r*lx+u*ly+l*lz local scr:
vector,vis:boolean=(camera::any):WorldToScreenPoint(w)if(vis)then projected=true
local x:number=scr.x local y:number=scr.y if(x<min_x)then min_x=x end if(y<min_y
)then min_y=y end if(x>max_x)then max_x=x end if(y>max_y)then max_y=y end end
end if(not projected)then return zero end return create(max(0,max_x-min_x),max(0
,max_y-min_y))end@native local function screen(udim:UDim__DARKLUA_TYPE_o,
reference:number):number return udim.Scale*reference+udim.Offset end local
Drawing={}::Drawing_module__DARKLUA_TYPE_x do function Drawing.attach(descriptor
:{[any]:{Link:Point__DARKLUA_TYPE_u?,From:Point__DARKLUA_TYPE_u?,To:
Point__DARKLUA_TYPE_u?,Size:UDim2__DARKLUA_TYPE_p?,Position:
UDim2__DARKLUA_TYPE_p?,AnchorPoint:vector?}}):Cluster__DARKLUA_TYPE_w assert(
type(descriptor)=='table',`Drawing.attach: expected table, got {type(descriptor)
}`)local attachments:{[any]:Attachment__DARKLUA_TYPE_v}={}local cluster=
setmetatable({Attachments=attachments,Active=true,Paused=false,Connection=nil},
Prototype)::any for object,config in descriptor do assert(type(config)=='table',
`Drawing.attach: expected config table, got {type(config)}`)assert(config.Link
or(config.From and config.To),`Drawing.attach: 'Link', or 'From' & 'To' are required.`
)attachments[object]={Link=config.Link,From=config.From,To=config.To,Size=config
.Size or DefaultSize,Position=config.Position or DefaultPosition,AnchorPoint=
config.AnchorPoint or DefaultAnchor}end@native local function Update():()if(not
cluster.Active or cluster.Paused)then return end local current_camera:any=(
workspace::any).CurrentCamera if(not current_camera)then return end local
viewport_size:vector=(current_camera::any).ViewportSize local cleanup:boolean=
true for draw_obj,attach in attachments do local link:any=attach.Link local from
:any=attach.From local to:any=attach.To if(typeof(link)=='Instance')then if((
link::Instance):IsA('Model')or(link::Instance):IsA('Folder'))then local cached=
ModelPointCache[link]if(not cached)then cached=PointModel.new(link::Instance)
ModelPointCache[link]=cached end link=cached attach.Link=cached end end local
is_line:boolean=from~=nil and to~=nil local destroyed:boolean=false if(type(link
)=='table'and not(link::any).Active)then destroyed=true end if(not destroyed)
then local inst:any=if(type(link)=='table')then rawget(link::any,'Instance')else
link if(inst and typeof(inst)=='Instance'and not(inst::Instance).Parent)then
destroyed=true end end if(is_line)then if(type(from)=='table'and not(from::any).
Active)then destroyed=true end if(type(to)=='table'and not(to::any).Active)then
destroyed=true end end if(destroyed)then(draw_obj::any).Visible=false continue
end cleanup=false if(is_line)then local w_from:vector?=nil local w_to:vector?=
nil local from_mt=if(type(from)=='table')then getmetatable(from::any)else nil
local to_mt=if(type(to)=='table')then getmetatable(to::any)else nil local
from_is_screen:boolean=from_mt==Point2D_mod local to_is_screen:boolean=to_mt==
Point2D_mod if(from_is_screen)then w_from=(from::any).Position elseif(type(from)
=='table')then local cf:any=(from::any).CFrame local ps:any=(from::any).Position
w_from=if(cf)then(cf::CFrame).Position elseif(ps)then(ps::vector)else nil elseif
(typeof(from)=='Instance')then if((from::Instance):IsA('BasePart'))then w_from=(
from::any).Position end end if(to_is_screen)then w_to=(to::any).Position elseif(
type(to)=='table')then local cf:any=(to::any).CFrame local ps:any=(to::any).
Position w_to=if(cf)then(cf::CFrame).Position elseif(ps)then(ps::vector)else nil
elseif(typeof(to)=='Instance')then if((to::Instance):IsA('BasePart'))then w_to=(
to::any).Position end end if(not w_from or not w_to)then(draw_obj::any).Visible=
false continue end local s_from:vector local s_to:vector local from_vis:boolean
local to_vis:boolean if(from_is_screen)then s_from=w_from::vector from_vis=true
else s_from,from_vis=(current_camera::any):WorldToScreenPoint(w_from::vector)end
if(to_is_screen)then s_to=w_to::vector to_vis=true else s_to,to_vis=(
current_camera::any):WorldToScreenPoint(w_to::vector)end if(not from_vis or not
to_vis)then(draw_obj::any).Visible=false continue end(draw_obj::any).From=s_from
;(draw_obj::any).To=s_to;(draw_obj::any).Visible=true else local link_is_screen:
boolean=if(type(link)=='table')then getmetatable(link::any)==Point2D_mod else
false local w_pos:vector?=nil local s_pos:vector?=nil local visible:boolean=
false if(link_is_screen)then w_pos=(link::any).Position s_pos=w_pos visible=true
elseif(type(link)=='table')then local cf:any=(link::any).CFrame local ps:any=(
link::any).Position w_pos=if(cf)then(cf::CFrame).Position elseif(ps)then(ps::
vector)else nil if(w_pos)then s_pos,visible=(current_camera::any):
WorldToScreenPoint(w_pos::vector)end elseif(typeof(link)=='Instance')then if((
link::Instance):IsA('BasePart'))then w_pos=(link::any).Position s_pos,visible=(
current_camera::any):WorldToScreenPoint(w_pos::vector)end end if(not w_pos or
not s_pos or not visible)then(draw_obj::any).Visible=false continue end local
projected_size:vector=zero local inst:any=if(type(link)=='table')then rawget(
link::any,'Instance')else link if(inst and typeof(inst)=='Instance'and(inst::
Instance):IsA('BasePart'))then projected_size=project((inst::any).CFrame,(inst::
any).Size,current_camera)elseif(type(link)=='table')then local cf:any=(link::any
).CFrame local sz:any=(link::any).Size if(cf and sz)then projected_size=project(
cf::CFrame,sz::vector,current_camera)elseif(link_is_screen)then projected_size=
zero end end local att_size:UDim2__DARKLUA_TYPE_p=attach.Size local att_pos:
UDim2__DARKLUA_TYPE_p=attach.Position local anchor:vector=attach.AnchorPoint
local width:number=screen(att_size.X,projected_size.x)local height:number=
screen(att_size.Y,projected_size.y);(draw_obj::any).Size=ceil(create(width,
height));(draw_obj::any).Position=ceil(create((s_pos::vector).x+screen(att_pos.X
,viewport_size.x)-(width*anchor.x),(s_pos::vector).y+screen(att_pos.Y,
viewport_size.y)-(height*anchor.y)));(draw_obj::any).Visible=true end end if(
cleanup)then(cluster::Cluster__DARKLUA_TYPE_w):Destroy()end end cluster.
Connection=(RunService::any).Render:Connect(Update)return cluster::any end end
return Drawing end function __DARKLUA_BUNDLE_MODULES.j():typeof(__modImpl())
local v=__DARKLUA_BUNDLE_MODULES.cache.j if not v then v={c=__modImpl()}
__DARKLUA_BUNDLE_MODULES.cache.j=v end return v.c end end end _G.bit64=
__DARKLUA_BUNDLE_MODULES.a()_G.fs=__DARKLUA_BUNDLE_MODULES.b()_G.http=
__DARKLUA_BUNDLE_MODULES.c()_G.signal=__DARKLUA_BUNDLE_MODULES.d()_G.UDim2=
__DARKLUA_BUNDLE_MODULES.e()_G.Point3D=__DARKLUA_BUNDLE_MODULES.f()_G.Point2D=
__DARKLUA_BUNDLE_MODULES.g()_G.PointInstance=__DARKLUA_BUNDLE_MODULES.h()_G.
PointModel=__DARKLUA_BUNDLE_MODULES.i()type EnumItem={EnumType:Enum,Value:number
,Name:string}type Enum={items:{[string]:EnumItem},name:string,insert:(string,
number)->Enum,GetEnumItems:()->{[string]:EnumItem},FromName:(string)->EnumItem?,
FromValue:(number)->EnumItem?}local render=__DARKLUA_BUNDLE_MODULES.j();(_G.
Drawing::any).attach=render.attach local EnumItem={}do local function 
constructor(name:string,value:number,parent:Enum):EnumItem assert('string'==
type(name),`bad argument #1 to EnumItem.new: string expected, got '{type(name)}'`
)assert('number'==type(value),`bad argument #2 to EnumItem.new: number expected, got '{
type(value)}'`)assert('table'==type(parent)and parent.items,`bad argument #3 to EnumItem.new: Enum expected, got '{
type(parent)}'`)local self=setmetatable({EnumType=parent,Value=value,Name=name},
EnumItem)parent.items[name]=self return self end EnumItem.new=constructor
function EnumItem:__tostring()return`Enum.{self.EnumType.name}.{self.Name}`end
EnumItem.__index=EnumItem end local Enums=setmetatable({GetEnums=function(self)
return self.items end,items={}},{__index=function(self,index)return self.items[
index]end})local Enum={}do local function constructor(name:string)assert(
'string'==type(name),`bad argument #1 to Enum.new: string expected, got '{type(
name)}'`)local self=setmetatable({items={},name=name},Enum)Enums.items[name]=
self return self end function Enum:GetEnumItems()return self.items end function
Enum:FromName(name:string)for _,Item:EnumItem in self.items do if Item.Name==
name then return Item end end return nil end function Enum:FromValue(value:
number)for _,Item:EnumItem in self.items do if Item.Value==value then return
Item end end return nil end function Enum:insert(name:string,value:number)
EnumItem.new(name,value,self)return self end Enum.new=constructor function Enum:
__tostring()return self.name end function Enum:__index(key)return self.items and
self.items[key]or rawget(Enum,key)end end Instance.declare{class='Instance',name
='Read',callback=function(self:Instance,offset:number,size:number):buffer return
memory.readbuffer(self,offset or 0,size or 32)end}Instance.declare{class=
'Instance',name='Write',callback=function(self:Instance,offset:number,data:
buffer):()return memory.writebuffer(self,offset or 0,data)end}task.delay(1,
function()local cache:string?=nil do if fs.file('spec.d.txt')then local ok,
content=pcall(fs.read,'spec.d.txt')if ok and type(content)=='string'then cache=
content:match('^%s*(%S+)%s*$')end end end local remote:string?=nil do local
response=http.get{url=
[[https://api.github.com/repos/flamingo300/roblox/contents/luau/spec.d.luau]]}if
type(response)=='string'and#response>0 then local ok,data=pcall(crypt.json.
decode,response)if ok and type(data)=='table'and type(data.sha)=='string'then
remote=data.sha end end end if remote and cache and remote==cache and fs.file(
'spec.d.luau')then return end if not remote and fs.file('spec.d.luau')then
return end local response=http.get{url=
[[https://raw.githubusercontent.com/flamingo300/roblox/master/luau/spec.d.luau]]
}if type(response)=='string'and#response>0 then fs.open('spec.d.luau'):write(
response):close()if remote then pcall(fs.write,'spec.d.txt',remote)end end end)
local API local time=os.clock()local function regenerate()local content=crypt.
json.decode(game:HttpGet(
[[https://raw.githubusercontent.com/MaximumADHD/Roblox-Client-Tracker/refs/heads/roblox/Full-API-Dump.json]]
))content.time=time fs.write('api.bin',crypt.base64.encode(crypt.json.encode(
content)))return content end if not fs.file('api.bin')then API=regenerate()else
local content=crypt.json.decode(crypt.base64.decode(fs.read('api.bin')))if
content.time<time-259200 then API=regenerate()else API=content end end for index
,data in API.Enums do local enum=Enum.new(data.Name)for _,item in data.Items do
enum:insert(item.Name,item.Value)end end Instance.declare{class='Instance',name=
'IsA',callback={method=function(self,className)local currentClass=self.ClassName
if currentClass==className then return true end for _,classData in API.Classes
do if classData.Name==currentClass then local superclass=classData.Superclass
while superclass and superclass~='<<<ROOT>>>'do if superclass==className then
return true end local found=false for _,parentClassData in API.Classes do if
parentClassData.Name==superclass then superclass=parentClassData.Superclass
found=true break end end if not found then break end end break end end return
false end}}_G.Enum=table.freeze(Enums)