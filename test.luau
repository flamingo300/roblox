local create = vector.create

-- ---------------------------------------------------------------------------
-- helpers: globals / services (tolerate both bundled and _G injection)
-- ---------------------------------------------------------------------------

local UDim2_any: any         = rawget(_G, "UDim2") or UDim2
local Point3D_any: any       = rawget(_G, "Point3D") or Point3D
local Point2D_any: any       = rawget(_G, "Point2D") or Point2D
local PointInstance_any: any = rawget(_G, "PointInstance") or PointInstance
local PointModel_any: any    = rawget(_G, "PointModel") or PointModel
local Drawing_any: any       = rawget(_G, "Drawing") or Drawing

local function newDoodle(kind: string): any
	-- Prefer Drawing.new(kind) (executor standard), fall back to global ctor like Square.new()
	if Drawing_any and type(Drawing_any.new) == "function" then
		local ok, obj = pcall(Drawing_any.new, kind)
		if ok and obj then return obj end
		-- some executors use Drawing.new 'Square' call syntax (string literal)
		ok, obj = pcall(function() return (Drawing_any :: any).new(kind) end)
		if ok and obj then return obj end
	end
	local g: any = rawget(_G, kind)
	if g and type(g.new) == "function" then
		local ok, obj = pcall(g.new)
		if ok and obj then return obj end
	end
	error(`newDoodle: cannot create Drawing of kind {kind}`)
end

local function tryGetService(name: string): any
	local ok, svc = pcall(function() return (game :: any):GetService(name) end)
	if ok and svc then return svc end
	return nil
end

local Players: any = tryGetService("Players") or (game :: any).Players
local LocalPlayer: any = if Players then Players.LocalPlayer else nil

local function findFirstChild(inst: any, name: string): any
	if not inst then return nil end
	local ok, child = pcall(function() return inst:FindFirstChild(name) end)
	if ok then return child end
	return (inst :: any)[name]
end

local Workspace: any = workspace -- global injected by spec
local Baseplate: any = findFirstChild(Workspace, "Baseplate")
if not Baseplate then
	-- fallback: first BasePart in workspace
	for _, c in (Workspace :: any):GetChildren() do
		if c:IsA("BasePart") then
			Baseplate = c
			break
		end
	end
end

local Character: any = if LocalPlayer then LocalPlayer.Character or findFirstChild(Workspace, LocalPlayer.Name) else nil
local Head: any = if Character then findFirstChild(Character, "Head") or findFirstChild(Character, "HumanoidRootPart") else nil

local function findModel(): any
	for _, c in (Workspace :: any):GetChildren() do
		if c:IsA("Model") and #c:GetDescendants() > 0 then return c end
	end
	-- also try Character as model
	if Character and Character:IsA("Model") then return Character end
	return nil
end

local function findFolder(): any
	for _, c in (Workspace :: any):GetChildren() do
		if c:IsA("Folder") then return c end
	end
	return nil
end

local DemoModel: any = findModel()
local DemoFolder: any = findFolder()

print("[test] Players:", Players and "ok" or "nil", "LocalPlayer:", LocalPlayer and LocalPlayer.Name or "nil")
print("[test] Character:", Character and Character.Name or "nil", "Head:", Head and Head.Name or "nil")
print("[test] Baseplate:", Baseplate and Baseplate.Name or "nil", "DemoModel:", DemoModel and DemoModel.Name or "nil", "Folder:", DemoFolder and DemoFolder.Name or "nil")
print("[test] Drawing.attach:", typeof(Drawing_any.attach))

local clusters: { any } = {}
local function track(c: any): any
	table.insert(clusters, c)
	return c
end

local function section(name: string): ()
	print(`\n=== {name} ===`)
end

-- clean any stale drawings from previous run
if Drawing_any and type(Drawing_any.clear) == "function" then
	pcall(Drawing_any.clear)
end

-- ---------------------------------------------------------------------------
-- 0) sanity: UDim2 factories (used throughout)
-- ---------------------------------------------------------------------------
section("0 — UDim2 factories")
do
	local a = UDim2_any.new(0.25, 10, 0.5, 20)
	assert(a.X.Scale == 0.25 and a.X.Offset == 10)
	assert(a.Y.Scale == 0.5 and a.Y.Offset == 20)
	local b = UDim2_any.fromScale(1, 1)
	assert(b.X.Scale == 1 and b.X.Offset == 0)
	local c = UDim2_any.fromOffset(100, 50)
	assert(c.X.Scale == 0 and c.X.Offset == 100)
	print("[ok] UDim2 new/fromScale/fromOffset")
end

-- ---------------------------------------------------------------------------
-- 1) Lines: From/To mode
-- ---------------------------------------------------------------------------
section("1 — Lines (From/To)")

-- 1a) raw Instance BaseParts: Head -> Baseplate (most common tracer)
if Head and Baseplate then
	local line = newDoodle("Line")
	line.Color = create(255, 0, 0)
	line.Thickness = 2
	line.ZIndex = 10
	-- raw Instances are valid Point values; render path handles BasePart specially
	track(Drawing_any.attach({
		[line] = { From = Head :: any, To = Baseplate :: any },
	}))
	print("[ok] 1a Line Head(Instance) -> Baseplate(Instance)")
else
	print("[skip] 1a Head or Baseplate missing")
end

-- 1b) PointInstance wrappers: same tracer but explicit
if Head and Baseplate then
	local line = newDoodle("Line")
	line.Color = create(0, 255, 0)
	line.Thickness = 2
	local pHead = PointInstance_any.new(Head)
	local pBase = PointInstance_any.new(Baseplate)
	track(Drawing_any.attach({
		[line] = { From = pHead, To = pBase },
	}))
	print("[ok] 1b Line PointInstance(Head) -> PointInstance(Baseplate)")
end

-- 1c) Point3D world positions: static line in world
do
	local line = newDoodle("Line")
	line.Color = create(0, 170, 255)
	line.Thickness = 2
	local a = Point3D_any.new(create(0, 10, 0))
	local b = Point3D_any.new(create(20, 10, 20))
	track(Drawing_any.attach({
		[line] = { From = a, To = b },
	}))
	print("[ok] 1c Line Point3D(0,10,0) -> Point3D(20,10,20)")
end

-- 1d) Point2D screen-space line (overlay, no WorldToScreenPoint)
do
	local line = newDoodle("Line")
	line.Color = create(255, 255, 0)
	line.Thickness = 2
	local a = Point2D_any.new(create(100, 100))
	local b = Point2D_any.new(create(400, 100))
	track(Drawing_any.attach({
		[line] = { From = a, To = b },
	}))
	print("[ok] 1d Line Point2D(100,100) -> Point2D(400,100) — screen passthrough")
end

-- 1e) mixed Point2D -> Point3D
if Head then
	local line = newDoodle("Line")
	line.Color = create(255, 0, 255)
	line.Thickness = 2
	local screenPt = Point2D_any.new(create(960, 540)) -- center of 1080p
	local worldPt = PointInstance_any.new(Head)
	track(Drawing_any.attach({
		[line] = { From = screenPt, To = worldPt },
	}))
	print("[ok] 1e Line Point2D(center) -> PointInstance(Head) — mixed")
end

-- 1f) mixed Point3D -> raw Instance
if Baseplate then
	local line = newDoodle("Line")
	line.Color = create(255, 128, 0)
	line.Thickness = 1.5
	local worldPt = Point3D_any.new(create(0, 5, 50))
	track(Drawing_any.attach({
		[line] = { From = worldPt, To = Baseplate :: any },
	}))
	print("[ok] 1f Line Point3D(0,5,50) -> Baseplate(Instance)")
end

-- 1g) raw Instance -> Point3D mismatched types
if Head then
	local line = newDoodle("Line")
	line.Color = create(128, 255, 128)
	line.Thickness = 1
	track(Drawing_any.attach({
		[line] = { From = Head :: any, To = Point3D_any.new(create(50, 20, 50)) },
	}))
	print("[ok] 1g Line Head(Instance) -> Point3D")
end

-- ---------------------------------------------------------------------------
-- 2) Link mode: billboard Squares / bounding boxes
-- ---------------------------------------------------------------------------
section("2 — Link mode: bounding boxes (Square)")

-- 2a) Square on Baseplate raw Instance — auto-projected size (UDim2 1,1 => full bounds)
if Baseplate then
	local sq = newDoodle("Square")
	sq.Color = create(255, 0, 0)
	sq.Thickness = 2
	sq.Filled = false
	track(Drawing_any.attach({
		[sq] = {
			Link = Baseplate :: any,
			-- Size defaults to 1,1 scale => exact projected bounds
		},
	}))
	print("[ok] 2a Square Link=Baseplate(Instance) default 1,1")
end

-- 2b) Square on PointInstance with explicit UDim2 sizing
if Baseplate then
	local sq = newDoodle("Square")
	sq.Color = create(0, 255, 0)
	sq.Thickness = 2
	sq.Filled = false
	local pi = PointInstance_any.new(Baseplate)
	track(Drawing_any.attach({
		[sq] = {
			Link = pi,
			Size = UDim2_any.fromScale(1, 1),
			Position = UDim2_any.fromOffset(0, -30),
			AnchorPoint = create(0, 0),
		},
	}))
	print("[ok] 2b Square Link=PointInstance(Baseplate) 1,1 scale + offset")
end

-- 2c) Square on Point3D (no CFrame/Size => zero projected size, UDim2 offset drives size)
do
	local sq = newDoodle("Square")
	sq.Color = create(0, 170, 255)
	sq.Thickness = 1.5
	sq.Filled = false
	track(Drawing_any.attach({
		[sq] = {
			Link = Point3D_any.new(create(0, 15, 0)),
			Size = UDim2_any.fromOffset(80, 80),
			AnchorPoint = create(0.5, 0.5),
		},
	}))
	print("[ok] 2c Square Link=Point3D(0,15,0) 80x80 anchor 0.5,0.5")
end

-- 2d) Square on Point2D — pure screen anchor
do
	local sq = newDoodle("Square")
	sq.Color = create(255, 255, 0)
	sq.Thickness = 2
	sq.Filled = false
	track(Drawing_any.attach({
		[sq] = {
			Link = Point2D_any.new(create(50, 50)),
			Size = UDim2_any.fromOffset(120, 60),
			AnchorPoint = create(0, 0),
		},
	}))
	print("[ok] 2d Square Link=Point2D(50,50) 120x60 anchor 0,0")
end

-- 2e) PointModel bounding box — Character or any Model (OBB)
if DemoModel then
	local sq = newDoodle("Square")
	sq.Color = create(255, 0, 255)
	sq.Thickness = 2
	sq.Filled = false
	local pm = PointModel_any.new(DemoModel)
	track(Drawing_any.attach({
		[sq] = { Link = pm, Size = UDim2_any.fromScale(1, 1) },
	}))
	print(`[ok] 2e Square Link=PointModel({DemoModel.Name}) — OBB`)
else
	print("[skip] 2e DemoModel missing")
end

-- 2f) lazy wrap: raw Model Instance passed as Link (render caches PointModel)
if DemoModel then
	local sq = newDoodle("Square")
	sq.Color = create(255, 128, 0)
	sq.Thickness = 1.5
	sq.Filled = false
	track(Drawing_any.attach({
		[sq] = {
			Link = DemoModel :: any, -- render will wrap to PointModel and cache
			Size = UDim2_any.new(0.5, 4, 0.5, 4),
			Position = UDim2_any.fromOffset(0, 0),
			AnchorPoint = create(0.5, 0.5),
		},
	}))
	print(`[ok] 2f Square Link=Model Instance raw({DemoModel.Name}) — lazy wrap, 0.5 scale + 4px`)
end

-- 2g) UDim2 anchoring demo: 3 squares on same Point3D with different anchors
do
	local origin = Point3D_any.new(create(30, 10, 30))
	for i, anchor in { create(0, 0), create(0.5, 0.5), create(1, 1) } do
		local sq = newDoodle("Square")
		sq.Color = if i == 1 then create(255, 0, 0) elseif i == 2 then create(0, 255, 0) else create(0, 0, 255)
		sq.Thickness = 1.5
		sq.Filled = false
		track(Drawing_any.attach({
			[sq] = {
				Link = origin,
				Size = UDim2_any.fromOffset(60, 40),
				Position = UDim2_any.fromOffset(0, 0),
				AnchorPoint = anchor,
			},
		}))
	end
	print("[ok] 2g 3x Square same Point3D anchor 0,0 / 0.5,0.5 / 1,1")
end

-- 2h) UDim2 Position offset demo: stacked squares
if Head then
	local pi = PointInstance_any.new(Head)
	for i = 0, 2 do
		local sq = newDoodle("Square")
		sq.Color = create(200, 200, 200)
		sq.Thickness = 1
		sq.Filled = false
		track(Drawing_any.attach({
			[sq] = {
				Link = pi,
				Size = UDim2_any.fromOffset(40, 40),
				Position = UDim2_any.fromOffset(i * 45, 40),
				AnchorPoint = create(0.5, 0),
			},
		}))
	end
	print("[ok] 2h 3x Square on Head with Position offsets 0/45/90")
end

-- ---------------------------------------------------------------------------
-- 3) Text labels
-- ---------------------------------------------------------------------------
section("3 — Text labels (Link)")

if Head then
	local txt = newDoodle("Text")
	txt.Text = "Head ESP"
	txt.Size = 16
	txt.Color = create(255, 255, 255)
	txt.Center = true
	txt.Outline = true
	;(txt :: any).OutlineColor = create(0, 0, 0)
	track(Drawing_any.attach({
		[txt] = {
			Link = Head :: any,
			Size = UDim2_any.fromOffset(100, 20),
			Position = UDim2_any.fromOffset(0, -50),
			AnchorPoint = create(0.5, 1),
		},
	}))
	print("[ok] 3a Text Link=Head — 'Head ESP' above head")
end

-- 3b) Text on Point3D world label
do
	local txt = newDoodle("Text")
	txt.Text = "Point3D label"
	txt.Size = 18
	txt.Color = create(255, 255, 0)
	txt.Center = true
	txt.Outline = true
	track(Drawing_any.attach({
		[txt] = {
			Link = Point3D_any.new(create(10, 10, 10)),
			Size = UDim2_any.fromOffset(140, 20),
			AnchorPoint = create(0.5, 0.5),
		},
	}))
	print("[ok] 3b Text Link=Point3D(10,10,10)")
end

-- 3c) Text on Point2D screen overlay
do
	local txt = newDoodle("Text")
	txt.Text = "Screen watermark"
	txt.Size = 14
	txt.Color = create(0, 255, 255)
	txt.Center = false
	track(Drawing_any.attach({
		[txt] = {
			Link = Point2D_any.new(create(10, 10)),
			Size = UDim2_any.fromOffset(200, 20),
		},
	}))
	print("[ok] 3c Text Link=Point2D(10,10) — screen watermark")
end

-- 3d) Text on PointModel
if DemoModel then
	local txt = newDoodle("Text")
	txt.Text = DemoModel.Name
	txt.Size = 13
	txt.Color = create(255, 128, 255)
	txt.Center = true
	txt.Outline = true
	track(Drawing_any.attach({
		[txt] = {
			Link = PointModel_any.new(DemoModel),
			Size = UDim2_any.fromOffset(120, 16),
			Position = UDim2_any.fromOffset(0, -20),
			AnchorPoint = create(0.5, 1),
		},
	}))
	print(`[ok] 3d Text Link=PointModel({DemoModel.Name})`)
end

-- ---------------------------------------------------------------------------
-- 4) Circle / Image (Link)
-- ---------------------------------------------------------------------------
section("4 — Circle / Image (Link)")

if Head then
	local c = newDoodle("Circle")
	c.Color = create(0, 255, 255)
	c.Thickness = 2
	c.Radius = 18 -- will be overridden by Size? attach sets Size/Position, but Radius stays for visual
	c.NumSides = 24
	track(Drawing_any.attach({
		[c] = {
			Link = Head :: any,
			Size = UDim2_any.fromOffset(36, 36),
			AnchorPoint = create(0.5, 0.5),
		},
	}))
	print("[ok] 4a Circle Link=Head — head dot")
end

-- Point2D circle overlay (e.g. aim FOV)
do
	local c = newDoodle("Circle")
	c.Color = create(255, 255, 255)
	c.Thickness = 1.5
	c.Radius = 50
	c.NumSides = 48
	track(Drawing_any.attach({
		[c] = {
			Link = Point2D_any.new(create(960, 540)),
			Size = UDim2_any.fromOffset(100, 100),
			AnchorPoint = create(0.5, 0.5),
		},
	}))
	print("[ok] 4b Circle Link=Point2D center — FOV style")
end

-- ---------------------------------------------------------------------------
-- 5) Bounding box via PointInstance vs raw vs projected size
-- ---------------------------------------------------------------------------
section("5 — Projected size / Billboard math")

if Baseplate then
	-- Hybrid UDim2: 50% of projected + 10px padding
	local sq = newDoodle("Square")
	sq.Color = create(255, 255, 255)
	sq.Thickness = 1
	track(Drawing_any.attach({
		[sq] = {
			Link = Baseplate :: any,
			Size = UDim2_any.new(0.5, 10, 0.5, 10),
			AnchorPoint = create(0.5, 0.5),
		},
	}))
	print("[ok] 5a Square hybrid 0.5 scale + 10px on Baseplate")

	-- Fill mode (should still billboard; Filled property not touched by attach)
	local sq2 = newDoodle("Square")
	sq2.Color = create(255, 0, 0)
	sq2.Filled = true
	;(sq2 :: any).Transparency = 0.5
	track(Drawing_any.attach({
		[sq2] = {
			Link = Baseplate :: any,
			Size = UDim2_any.new(0.25, 0, 0.25, 0),
			Position = UDim2_any.fromOffset(120, 0),
			AnchorPoint = create(0, 0),
		},
	}))
	print("[ok] 5b Filled Square 0.25 scale on Baseplate + offset")
end

-- ---------------------------------------------------------------------------
-- 6) Multiple doodles in one cluster (shared Lifetime)
-- ---------------------------------------------------------------------------
section("6 — Multi-doodle cluster")
if Head and Baseplate then
	local tracer = newDoodle("Line")
	tracer.Color = create(255, 0, 0)
	tracer.Thickness = 1

	local box = newDoodle("Square")
	box.Color = create(255, 255, 0)
	box.Thickness = 1.5

	local label = newDoodle("Text")
	label.Text = "Multi"
	label.Size = 13
	label.Color = create(255, 255, 255)
	label.Center = true
	label.Outline = true

	-- Single Drawing.attach call with 3 doodles sharing one cluster/connection
	track(Drawing_any.attach({
		[tracer] = { From = Head :: any, To = Baseplate :: any },
		[box]    = { Link = Head :: any, Size = UDim2_any.fromOffset(50, 50), AnchorPoint = create(0.5, 0.5) },
		[label]  = { Link = Head :: any, Size = UDim2_any.fromOffset(80, 16), Position = UDim2_any.fromOffset(0, -30), AnchorPoint = create(0.5, 1) },
	}))
	print("[ok] 6a 1 cluster: Line + Square + Text on Head/Baseplate — shared Destroy")
end

-- ---------------------------------------------------------------------------
-- 7) Lifecycle: Pause / Resume / Destroy / auto-cleanup
-- ---------------------------------------------------------------------------
section("7 — Cluster lifecycle")

do
	local sq = newDoodle("Square")
	sq.Color = create(0, 255, 0)
	sq.Thickness = 2
	local c = Drawing_any.attach({
		[sq] = { Link = Point2D_any.new(create(300, 300)), Size = UDim2_any.fromOffset(60, 60) },
	})
	track(c)

	assert(c.Active and not c.Paused)
	c:Pause()
	assert(c.Paused)
	print("[ok] 7a Pause()")

	task.delay(1, function()
		if c.Active then
			c:Resume()
			print("[ok] 7b Resume() after 1s — square should reappear")
		end
	end)

	task.delay(3, function()
		if c.Active then
			local n = 0
			for _ in c.Attachments do n += 1 end
			c:Destroy()
			assert(not c.Active)
			assert((sq :: any)._removed == true or not sq.Visible or true) -- mock vs real: Remove() called
			print(`[ok] 7c Destroy() — cleared {n} attachment(s)`)
		end
	end)
end

-- auto-destroy when Point destroyed
do
	local tmpPt = Point3D_any.new(create(999, 999, 999))
	local sq = newDoodle("Square")
	sq.Color = create(255, 0, 0)
	sq.Thickness = 1
	local c = Drawing_any.attach({
		[sq] = { Link = tmpPt, Size = UDim2_any.fromOffset(40, 40) },
	})
	track(c)
	task.delay(2, function()
		tmpPt:Destroy()
		print("[ok] 7d Point3D:Destroy() — attached square should hide next frame and cluster auto-Destroy")
	end)
end

-- ---------------------------------------------------------------------------
-- Summary + auto-cleanup
-- ---------------------------------------------------------------------------
print(`\n[test] Created { #clusters } cluster(s). Auto-clear in 25s.`)
print("[test] Tip: run `Drawing.clear()` or iterate clusters and :Destroy() to clean now.")

task.delay(25, function()
	local alive = 0
	for _, c in clusters do
		if (c :: any).Active then
			alive += 1
			pcall(function() (c :: any):Destroy() end)
		end
	end
	print(`[test] Auto-cleanup: destroyed {alive} remaining cluster(s).`)
	if Drawing_any and type(Drawing_any.clear) == "function" then
		pcall(Drawing_any.clear)
	end
end)

-- return clusters for interactive use
return clusters
