import { createClient } from '@supabase/supabase-js';

const supabaseURL = "https://nintqyjbyfonoiwrjkvp.supabase.co";
const supabaseAnonKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5pbnRxeWpieWZvbm9pd3Jqa3ZwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUyOTE4MTMsImV4cCI6MjA3MDg2NzgxM30.LToCeCGqaWmEvFQLlWMxTH2EFaek2fqcnhN0qUT60xA"

const supabase = createClient(supabaseURL, supabaseAnonKey);

export default supabase;    