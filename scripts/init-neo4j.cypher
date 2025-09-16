// Neo4j initialization script for Clario
// Creates constraints, indexes, and sample data structure

// Node uniqueness constraints
CREATE CONSTRAINT meeting_id IF NOT EXISTS FOR (m:Meeting) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT organization_id IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT outcome_id IF NOT EXISTS FOR (out:Outcome) REQUIRE out.id IS UNIQUE;

// Performance indexes
CREATE INDEX meeting_date IF NOT EXISTS FOR (m:Meeting) ON (m.scheduled_at);
CREATE INDEX decision_date IF NOT EXISTS FOR (d:Decision) ON (d.decided_at);
CREATE INDEX decision_type IF NOT EXISTS FOR (d:Decision) ON (d.decision_type);
CREATE INDEX decision_status IF NOT EXISTS FOR (d:Decision) ON (d.status);
CREATE INDEX topic_category IF NOT EXISTS FOR (t:Topic) ON (t.category);
CREATE INDEX outcome_status IF NOT EXISTS FOR (o:Outcome) ON (o.status);
CREATE INDEX organization_filter IF NOT EXISTS FOR (n) ON (n.organization_id);

// Composite indexes for common queries
CREATE INDEX meeting_org_date IF NOT EXISTS FOR (m:Meeting) ON (m.organization_id, m.scheduled_at);
CREATE INDEX decision_org_type IF NOT EXISTS FOR (d:Decision) ON (d.organization_id, d.decision_type);

// Sample nodes to demonstrate the structure (will be replaced by real data)
CREATE (org:Organization {
  id: 'sample_org_001',
  name: 'Sample Organization',
  description: 'A sample organization for Clario demo',
  created_at: datetime()
});

CREATE (topic1:Topic {
  name: 'Budget Planning',
  category: 'financial',
  frequency: 0,
  created_at: datetime()
});

CREATE (topic2:Topic {
  name: 'Team Scaling',
  category: 'personnel',
  frequency: 0,
  created_at: datetime()
});

CREATE (topic3:Topic {
  name: 'Product Strategy',
  category: 'strategic',
  frequency: 0,
  created_at: datetime()
});

// Sample meeting
CREATE (meeting1:Meeting {
  id: 'sample_meeting_001',
  title: 'Q4 Planning Session',
  description: 'Planning for Q4 goals and budget allocation',
  meeting_type: 'planning',
  status: 'completed',
  organization_id: 'sample_org_001',
  scheduled_at: datetime('2024-01-01T09:00:00Z'),
  duration_minutes: 90,
  confidence_score: 0.85,
  created_at: datetime()
});

// Sample decision
CREATE (decision1:Decision {
  id: 'sample_decision_001',
  title: 'Increase Q4 Marketing Budget',
  description: 'Allocate additional $50K to marketing for Q4 campaigns',
  decision_type: 'financial',
  urgency: 'medium',
  status: 'approved',
  reasoning: 'Market analysis shows opportunity for increased ROI in Q4',
  estimated_cost: 50000,
  timeline: '30 days',
  confidence_score: 0.78,
  meeting_id: 'sample_meeting_001',
  organization_id: 'sample_org_001',
  decided_at: datetime('2024-01-01T10:30:00Z'),
  implementation_progress: 75,
  success_rating: 0.8,
  created_at: datetime()
});

// Sample outcome
CREATE (outcome1:Outcome {
  id: 'sample_outcome_001',
  decision_id: 'sample_decision_001',
  description: 'Marketing campaign generated 25% increase in leads',
  outcome_type: 'result',
  status: 'completed',
  success_score: 0.85,
  lessons_learned: 'Q4 timing was optimal for increased marketing spend',
  occurred_at: datetime('2024-02-01T00:00:00Z'),
  created_at: datetime()
});

// Create relationships
MATCH (m:Meeting {id: 'sample_meeting_001'}), (d:Decision {id: 'sample_decision_001'})
MERGE (m)-[r1:RESULTED_IN]->(d)
SET r1.created_at = datetime();

MATCH (m:Meeting {id: 'sample_meeting_001'}), (t:Topic {name: 'Budget Planning'})
MERGE (m)-[r2:DISCUSSED]->(t)
SET r2.relevance = 0.9, r2.created_at = datetime();

MATCH (d:Decision {id: 'sample_decision_001'}), (t:Topic {name: 'Budget Planning'})
MERGE (d)-[r3:AFFECTS]->(t)
SET r3.impact = 0.8, r3.created_at = datetime();

MATCH (d:Decision {id: 'sample_decision_001'}), (o:Outcome {id: 'sample_outcome_001'})
MERGE (d)-[r4:LED_TO]->(o)
SET r4.created_at = datetime();

// Create some additional sample data for pattern analysis
CREATE (meeting2:Meeting {
  id: 'sample_meeting_002',
  title: 'Engineering Team Expansion',
  description: 'Discussing hiring plan for engineering team',
  meeting_type: 'decision',
  status: 'completed',
  organization_id: 'sample_org_001',
  scheduled_at: datetime('2024-01-15T14:00:00Z'),
  duration_minutes: 60,
  confidence_score: 0.92,
  created_at: datetime()
});

CREATE (decision2:Decision {
  id: 'sample_decision_002',
  title: 'Hire 3 Senior Engineers',
  description: 'Expand engineering team by hiring 3 senior engineers',
  decision_type: 'personnel',
  urgency: 'high',
  status: 'approved',
  reasoning: 'Need to accelerate product development for Q2 launch',
  estimated_cost: 180000,
  timeline: '60 days',
  confidence_score: 0.88,
  meeting_id: 'sample_meeting_002',
  organization_id: 'sample_org_001',
  decided_at: datetime('2024-01-15T15:00:00Z'),
  implementation_progress: 90,
  success_rating: 0.75,
  created_at: datetime()
});

// Connect the new meeting and decision
MATCH (m:Meeting {id: 'sample_meeting_002'}), (d:Decision {id: 'sample_decision_002'})
MERGE (m)-[r:RESULTED_IN]->(d)
SET r.created_at = datetime();

MATCH (m:Meeting {id: 'sample_meeting_002'}), (t:Topic {name: 'Team Scaling'})
MERGE (m)-[r:DISCUSSED]->(t)
SET r.relevance = 0.95, r.created_at = datetime();

MATCH (d:Decision {id: 'sample_decision_002'}), (t:Topic {name: 'Team Scaling'})
MERGE (d)-[r:AFFECTS]->(t)
SET r.impact = 0.9, r.created_at = datetime();

// Create temporal relationship between meetings
MATCH (m1:Meeting {id: 'sample_meeting_001'}), (m2:Meeting {id: 'sample_meeting_002'})
MERGE (m1)-[r:PRECEDED]->(m2)
SET r.time_gap_hours = duration.between(m1.scheduled_at, m2.scheduled_at).hours;

// Create decision dependency (hiring depends on budget approval)
MATCH (d1:Decision {id: 'sample_decision_001'}), (d2:Decision {id: 'sample_decision_002'})
MERGE (d2)-[r:DEPENDS_ON]->(d1)
SET r.dependency_type = 'budget_prerequisite', r.created_at = datetime();
