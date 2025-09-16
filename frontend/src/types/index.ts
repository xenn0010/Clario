export interface AgentSummary {
  agent_id: string;
  name: string;
  personality: string;
  role: string;
  status: string;
  expertise_areas: string[];
}

export interface MeetingPulse {
  meeting_id: string;
  title: string;
  status: 'scheduled' | 'in_progress' | 'completed';
  scheduled_for: string;
  participants: string[];
}

export interface AgentInteraction {
  author: string;
  sentiment: 'optimistic' | 'cautious' | 'critical';
  content: string;
  timestamp: string;
  action?: string;
}

export interface GraphNode {
  id: string;
  label: string;
  type: 'agent' | 'meeting' | 'decision' | 'topic' | 'outcome';
  activity?: number;
  sentiment?: 'positive' | 'neutral' | 'warning';
}

export interface GraphLink {
  source: string;
  target: string;
  intensity?: number;
  label?: string;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface GraphMetric {
  name: string;
  value: string;
  delta?: number;
  tone?: 'positive' | 'neutral' | 'negative';
}

export interface OrganizationPulse {
  graph: GraphData;
  metrics: GraphMetric[];
}
