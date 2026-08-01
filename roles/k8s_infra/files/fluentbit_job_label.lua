-- Builds promtail's `job` label: "<namespace>/<pod>".
--
-- WHY A LUA FILTER AND NOT A RECORD ACCESSOR: out_loki's `labels` option does
-- support record accessors, but ONLY a single accessor per value. A pattern
-- mixing a literal with two accessors, i.e.
--     labels job=$kubernetes['namespace_name']/$kubernetes['pod_name']
-- is rejected by flb_ra_create() at startup:
--     [record accessor] bad input character '/' at line 0
--     [record accessor] syntax error, unexpected '$', expecting end of file
--     [output:loki] invalid record accessor pattern for key 'job'
-- and fluent-bit then SIGSEGVs in flb_loki_kv_destroy() on that error path
-- (upstream bug, the plugin frees a partially built kv). Verified on
-- fluent-bit 4.2.7. Promtail builds the same label with a relabel concat
-- (separator '/'), which has no equivalent in the loki output plugin, so the
-- concatenation has to happen on the record BEFORE it reaches the output.
--
-- The `job` key is then consumed by `labels job=$job` and removed from the log
-- line automatically (out_loki appends every label accessor to its internal
-- remove_keys list).
--
-- Return code 2 = record modified, keep the original timestamp.
function set_job(tag, ts, record)
    local k = record["kubernetes"]
    if k ~= nil and k["namespace_name"] ~= nil and k["pod_name"] ~= nil then
        record["job"] = k["namespace_name"] .. "/" .. k["pod_name"]
        return 2, ts, record
    end
    -- No kubernetes metadata (should not happen for kube.* records): leave the
    -- record untouched rather than emitting a half-built label.
    return 0, ts, record
end
