-- Gridlock 2.0 — initial schema
-- Postgres / Supabase. No authentication; auditability via hash-chained evidence_audit.

create extension if not exists "pgcrypto";   -- gen_random_uuid()

-- ============================================================
-- Enums (closed sets)
-- ============================================================
create type violation_type  as enum ('helmet','triple_riding','seatbelt','wrong_side','stop_line','red_light','illegal_parking');
create type confidence_band as enum ('auto_confirm','human_review','discard');
create type violation_status as enum ('pending','confirmed','rejected','challan_issued');
create type job_status      as enum ('queued','processing','done','failed');
create type media_type      as enum ('image','video','frame');
create type review_action   as enum ('confirm','reject','escalate');

-- ============================================================
-- cameras — a source (physical camera or the "Uploads" pseudo-camera).
-- Holds the per-camera Scene Context Model (geometry annotated once).
-- ============================================================
create table cameras (
  id             uuid primary key default gen_random_uuid(),
  name           text not null,
  location_name  text,
  latitude       double precision,
  longitude      double precision,
  scene_context  jsonb not null default '{}'::jsonb,  -- stop_line, no_park polys, lane vectors, signal ROI, homography
  is_active      boolean not null default true,
  created_at     timestamptz not null default now()
);

-- ============================================================
-- ingestion_jobs — one uploaded image/video and its processing lifecycle.
-- ============================================================
create table ingestion_jobs (
  id             uuid primary key default gen_random_uuid(),
  camera_id      uuid references cameras(id) on delete set null,
  media_type     media_type not null default 'image',
  storage_path   text not null,                 -- path in the 'originals' bucket
  sha256         text,
  captured_at    timestamptz,
  uploaded_at    timestamptz not null default now(),
  status         job_status not null default 'queued',
  processing_ms  integer,
  error          text,
  width          integer,
  height         integer,
  frame_count    integer,                        -- for video
  model_version  text,
  created_at     timestamptz not null default now()
);

-- ============================================================
-- detections — every object found in a job (vehicle / person / plate / light).
-- ============================================================
create table detections (
  id           uuid primary key default gen_random_uuid(),
  job_id       uuid not null references ingestion_jobs(id) on delete cascade,
  frame_index  integer,                          -- for video
  class_label  text not null,
  confidence   real,
  bbox         jsonb not null,                   -- {x,y,w,h}
  track_id     integer,                          -- persistent id across frames (video)
  attributes   jsonb not null default '{}'::jsonb,
  created_at   timestamptz not null default now()
);

-- ============================================================
-- plates — ANPR results, normalized for repeat-offender lookups.
-- ============================================================
create table plates (
  id               uuid primary key default gen_random_uuid(),
  plate_text       text not null,                -- raw OCR
  plate_normalized text not null unique,         -- uppercased, stripped
  state_code       text,
  is_valid_format  boolean,
  ocr_confidence   real,
  first_seen_at    timestamptz not null default now(),
  last_seen_at     timestamptz not null default now(),
  created_at       timestamptz not null default now()
);

-- ============================================================
-- violations — the core record (one per detected violation).
-- ============================================================
create table violations (
  id                   uuid primary key default gen_random_uuid(),
  job_id               uuid not null references ingestion_jobs(id) on delete cascade,
  camera_id            uuid references cameras(id) on delete set null,
  violation_type       violation_type not null,
  confidence           real,
  confidence_band      confidence_band not null default 'human_review',
  status               violation_status not null default 'pending',
  detected_at          timestamptz not null default now(),
  annotated_image_path text,                     -- path in the 'annotated' bucket
  evidence             jsonb not null default '{}'::jsonb,  -- bboxes, involved detection ids
  vlm_caption          text,
  model_module         text,
  model_version        text,
  sha256_evidence      text,
  signature            text,
  plate_id             uuid references plates(id) on delete set null,
  metadata             jsonb not null default '{}'::jsonb,
  created_at           timestamptz not null default now(),
  updated_at           timestamptz not null default now()
);

-- ============================================================
-- review_actions — human review-band decisions (anonymous; no auth).
-- ============================================================
create table review_actions (
  id             uuid primary key default gen_random_uuid(),
  violation_id   uuid not null references violations(id) on delete cascade,
  action         review_action not null,
  reviewer_label text,
  notes          text,
  created_at     timestamptz not null default now()
);

-- ============================================================
-- evidence_audit — append-only, hash-chained tamper-evident trail.
-- sha256 = H(payload || prev_hash) ; prev_hash = previous row's sha256 for this violation.
-- ============================================================
create table evidence_audit (
  id           uuid primary key default gen_random_uuid(),
  violation_id uuid not null references violations(id) on delete cascade,
  event_type   text not null,                    -- created | reviewed | exported | hash_verified
  payload      jsonb not null default '{}'::jsonb,
  sha256       text not null,
  prev_hash    text,
  created_at   timestamptz not null default now()
);

-- ============================================================
-- model_registry — which model/version produced results (no silent substitution).
-- ============================================================
create table model_registry (
  id           uuid primary key default gen_random_uuid(),
  module       text not null,
  model_name   text not null,
  variant      text,
  version      text,
  metric_name  text,
  metric_value real,
  checkpoint_ref text,
  is_active    boolean not null default true,
  deployed_at  timestamptz not null default now()
);

-- ============================================================
-- Indexes
-- ============================================================
create index idx_violations_type        on violations(violation_type);
create index idx_violations_detected_at on violations(detected_at desc);
create index idx_violations_camera      on violations(camera_id);
create index idx_violations_plate       on violations(plate_id);
create index idx_violations_band_status on violations(confidence_band, status);
create index idx_detections_job         on detections(job_id);
create index idx_jobs_status_created    on ingestion_jobs(status, created_at desc);

-- ============================================================
-- updated_at trigger on violations
-- ============================================================
create or replace function set_updated_at() returns trigger as $$
begin new.updated_at = now(); return new; end;
$$ language plpgsql;

create trigger trg_violations_updated
  before update on violations
  for each row execute function set_updated_at();

-- ============================================================
-- Analytics views
-- ============================================================
create view v_violations_by_type as
  select violation_type,
         count(*)                                          as total,
         count(*) filter (where status = 'confirmed')      as confirmed
  from violations group by violation_type;

create view v_violations_by_day as
  select date_trunc('day', detected_at) as day, violation_type, count(*) as total
  from violations group by 1, 2 order by 1;

create view v_violations_by_camera as
  select c.id as camera_id, c.name, count(v.*) as total
  from cameras c left join violations v on v.camera_id = c.id
  group by c.id, c.name;

create view v_hotspots as
  select c.id as camera_id, c.name, c.latitude, c.longitude, count(v.*) as total
  from cameras c join violations v on v.camera_id = c.id
  where c.latitude is not null
  group by c.id, c.name, c.latitude, c.longitude;

create view v_repeat_offenders as
  select p.id as plate_id, p.plate_normalized, p.state_code,
         count(v.*) as violation_count, max(v.detected_at) as last_seen
  from plates p join violations v on v.plate_id = p.id
  group by p.id, p.plate_normalized, p.state_code
  having count(v.*) > 1
  order by violation_count desc;

create view v_confidence_bands as
  select confidence_band, count(*) as total from violations group by confidence_band;
