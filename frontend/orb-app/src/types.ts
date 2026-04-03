export type Message = { role: 'user' | 'ai'; text: string; isDiff?: boolean; imageData?: string };
export type OrbMode = 'idle' | 'querying' | 'agent-processing';
export type ActiveMode = 'query' | 'agent' | 'tutor' | 'imagegen';
export type TestCase = { id: number; input: string; expected: string };
export type TestResult = { input: string; expected: string; actual: string | null; passed: boolean; error: string | null };
export type ChatSession = { session_id: string; title: string; created_at?: string; turn_count?: number };

export type CharEntry = { lora: string; description: string; position: string; outfit: string };

export type LoraItem = {
  name: string;
  filename: string;
  path: string;
  size_mb: number;
  trigger_words: string[];
  preview_image?: string | null;
};

export type GamifProfile = {
  xp: number;
  level: number;
  level_progress: { progress_pct: number; xp_for_next: number; current_level_xp: number };
  streak_days: number;
  total_solved: number;
  total_attempted: number;
  badges: { emoji: string; name: string; earned_at: string }[];
  topic_streaks: Record<string, number>;
};

export const API_BASE = 'http://localhost:8000';
