import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const anon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

// Browser client — anon key only. Reads are allowed by RLS; writes go through the backend.
export const supabase = createClient(url, anon);

export function storageUrl(bucket: string, path: string | null): string | null {
  if (!path) return null;
  return `${url}/storage/v1/object/public/${bucket}/${path}`;
}
