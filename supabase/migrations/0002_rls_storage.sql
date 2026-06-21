-- Gridlock 2.0 — RLS + storage buckets
-- No auth: the anon key is READ-ONLY; all writes go through the backend using the
-- service_role key (which bypasses RLS). This keeps the public demo safe-ish.

-- ============================================================
-- Enable RLS + public read-only policies (anon + authenticated may SELECT)
-- ============================================================
do $$
declare t text;
begin
  foreach t in array array[
    'cameras','ingestion_jobs','detections','plates',
    'violations','review_actions','evidence_audit','model_registry'
  ] loop
    execute format('alter table %I enable row level security;', t);
    execute format(
      'create policy %I on %I for select to anon, authenticated using (true);',
      'public_read_' || t, t
    );
  end loop;
end $$;

-- Writes: no anon policies are created, so anon cannot INSERT/UPDATE/DELETE.
-- The backend uses the service_role key, which bypasses RLS entirely.

-- ============================================================
-- Storage buckets (public read; writes via service_role)
-- ============================================================
insert into storage.buckets (id, name, public) values
  ('originals', 'originals', true),
  ('annotated', 'annotated', true),
  ('crops',     'crops',     true)
on conflict (id) do nothing;
