pcall(table.clear, _G)
for i,v in _G do
    _G[i] = nil
end

table.clear(getfenv())
