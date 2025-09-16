"""
Decision flow analysis and pattern recognition for Clario
Analyzes decision chains, consequences, and organizational patterns
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from app.services.graph.neo4j_service import get_neo4j_service
from app.core.logging import get_logger

logger = get_logger("decision_flow")


class DecisionFlowType(str, Enum):
    """Types of decision flows"""
    LINEAR = "linear"  # A → B → C
    BRANCHING = "branching"  # A → B, A → C
    CONVERGING = "converging"  # A → C, B → C
    CIRCULAR = "circular"  # A → B → C → A


class OutcomeType(str, Enum):
    """Types of decision outcomes"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass
class DecisionNode:
    """Decision node in flow analysis"""
    id: str
    title: str
    decision_type: str
    status: str
    success_rating: float
    decided_at: datetime
    implementation_progress: int
    topics: List[str]
    outcomes: List[Dict[str, Any]]


@dataclass
class DecisionFlow:
    """Complete decision flow structure"""
    flow_id: str
    flow_type: DecisionFlowType
    decisions: List[DecisionNode]
    relationships: List[Dict[str, Any]]
    total_success_rate: float
    duration_days: int
    key_topics: List[str]
    lessons_learned: List[str]


class DecisionFlowAnalyzer:
    """Analyzes decision flows and organizational patterns"""
    
    def __init__(self):
        self.neo4j_service = None
        
    async def initialize(self) -> None:
        """Initialize the analyzer"""
        self.neo4j_service = await get_neo4j_service()
        logger.info("Decision flow analyzer initialized")
    
    async def analyze_decision_consequences(
        self, 
        decision_id: str
    ) -> Dict[str, Any]:
        """Analyze the complete consequences of a decision"""
        try:
            # Get the decision flow from Neo4j
            flow_data = await self.neo4j_service.analyze_decision_flow(decision_id)
            
            if not flow_data:
                return {"error": "Decision not found"}
            
            decision = flow_data.get("decision", {})
            outcomes = flow_data.get("outcomes", [])
            influenced_decisions = flow_data.get("influenced_decisions", [])
            affected_topics = flow_data.get("affected_topics", [])
            
            # Calculate consequence metrics
            consequence_analysis = {
                "decision_id": decision_id,
                "decision_title": decision.get("title", ""),
                "decision_type": decision.get("decision_type", ""),
                "direct_outcomes": self._analyze_direct_outcomes(outcomes),
                "cascade_effects": self._analyze_cascade_effects(influenced_decisions),
                "topic_impact": self._analyze_topic_impact(affected_topics),
                "success_metrics": self._calculate_success_metrics(decision, outcomes),
                "timeline_analysis": self._analyze_timeline(decision, outcomes, influenced_decisions),
                "risk_factors": self._identify_risk_factors(decision, outcomes),
                "lessons_learned": self._extract_lessons_learned(outcomes)
            }
            
            return consequence_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze decision consequences: {e}")
            return {"error": str(e)}
    
    async def map_organizational_decision_patterns(
        self, 
        organization_id: str,
        time_period_days: int = 180
    ) -> Dict[str, Any]:
        """Map organizational decision-making patterns"""
        try:
            # Get similar decision patterns
            patterns = {}
            
            # Analyze by decision type
            decision_types = ["strategic", "operational", "financial", "technical", "personnel"]
            
            for decision_type in decision_types:
                type_patterns = await self.neo4j_service.find_similar_decision_patterns(
                    decision_type=decision_type,
                    organization_id=organization_id,
                    limit=20
                )
                
                patterns[decision_type] = {
                    "total_decisions": len(type_patterns),
                    "average_success": self._calculate_average_success(type_patterns),
                    "common_topics": self._extract_common_topics(type_patterns),
                    "success_factors": self._identify_success_factors(type_patterns),
                    "failure_patterns": self._identify_failure_patterns(type_patterns)
                }
            
            # Get topic trends
            topic_trends = await self.neo4j_service.get_topic_trends(
                organization_id=organization_id,
                days=time_period_days
            )
            
            # Analyze decision success patterns
            success_analysis = await self.neo4j_service.analyze_decision_success_patterns(
                organization_id
            )
            
            return {
                "organization_id": organization_id,
                "analysis_period_days": time_period_days,
                "decision_type_patterns": patterns,
                "topic_trends": topic_trends,
                "success_patterns": success_analysis,
                "organizational_insights": self._generate_organizational_insights(patterns, topic_trends),
                "recommendations": self._generate_recommendations(patterns, success_analysis)
            }
            
        except Exception as e:
            logger.error(f"Failed to map organizational patterns: {e}")
            return {"error": str(e)}
    
    async def predict_decision_outcome(
        self, 
        decision_context: Dict[str, Any],
        organization_id: str
    ) -> Dict[str, Any]:
        """Predict likely outcomes based on historical patterns"""
        try:
            decision_type = decision_context.get("decision_type", "operational")
            topics = decision_context.get("topics", [])
            
            # Find similar historical decisions
            similar_patterns = await self.neo4j_service.find_similar_decision_patterns(
                decision_type=decision_type,
                organization_id=organization_id,
                limit=10
            )
            
            if not similar_patterns:
                return {
                    "prediction": "insufficient_data",
                    "confidence": 0.0,
                    "message": "Not enough historical data for prediction"
                }
            
            # Analyze patterns for prediction
            success_rates = [p["average_success"] for p in similar_patterns if p["average_success"]]
            topic_matches = self._calculate_topic_similarity(topics, similar_patterns)
            
            # Calculate prediction
            base_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.5
            topic_adjustment = topic_matches * 0.2  # Boost for topic similarity
            
            predicted_success = min(1.0, base_success_rate + topic_adjustment)
            confidence = len(similar_patterns) / 10.0  # Higher confidence with more data
            
            # Identify potential risks and opportunities
            risks = self._identify_potential_risks(similar_patterns, decision_context)
            opportunities = self._identify_opportunities(similar_patterns, decision_context)
            
            return {
                "prediction": {
                    "success_probability": predicted_success,
                    "confidence_level": confidence,
                    "expected_duration": self._estimate_duration(similar_patterns),
                    "risk_level": self._assess_risk_level(risks)
                },
                "similar_decisions": len(similar_patterns),
                "historical_success_rate": base_success_rate,
                "topic_similarity_score": topic_matches,
                "potential_risks": risks,
                "opportunities": opportunities,
                "recommendations": self._generate_decision_recommendations(
                    similar_patterns, risks, opportunities
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to predict decision outcome: {e}")
            return {"error": str(e)}
    
    async def trace_decision_chain(
        self, 
        starting_decision_id: str
    ) -> Dict[str, Any]:
        """Trace the complete chain of decisions that followed"""
        try:
            chain_data = await self.neo4j_service.get_decision_chain(starting_decision_id)
            
            if not chain_data:
                return {"error": "Decision chain not found"}
            
            starting_decision = chain_data.get("starting_decision", {})
            future_decisions = chain_data.get("future_decisions", [])
            
            # Analyze the chain
            chain_analysis = {
                "starting_decision": starting_decision,
                "chain_length": len(future_decisions),
                "total_duration": self._calculate_chain_duration(starting_decision, future_decisions),
                "cascade_success_rate": self._calculate_cascade_success_rate(future_decisions),
                "decision_types_involved": list(set(d.get("decision_type", "") for d in future_decisions)),
                "topics_evolution": self._analyze_topic_evolution(starting_decision, future_decisions),
                "decision_velocity": self._calculate_decision_velocity(future_decisions),
                "impact_amplification": self._analyze_impact_amplification(starting_decision, future_decisions)
            }
            
            return chain_analysis
            
        except Exception as e:
            logger.error(f"Failed to trace decision chain: {e}")
            return {"error": str(e)}
    
    async def find_decision_bottlenecks(
        self, 
        organization_id: str
    ) -> List[Dict[str, Any]]:
        """Identify bottlenecks in organizational decision-making"""
        try:
            # Query for decisions with long implementation times
            query = """
            MATCH (d:Decision)
            WHERE d.organization_id = $organization_id
              AND d.implementation_progress < 100
              AND d.decided_at < datetime() - duration('P30D')
            OPTIONAL MATCH (d)-[:AFFECTS]->(t:Topic)
            OPTIONAL MATCH (blocking:Decision)-[:DEPENDS_ON]->(d)
            
            RETURN d as decision,
                   collect(DISTINCT t.name) as topics,
                   count(blocking) as blocked_decisions,
                   duration.between(d.decided_at, datetime()).days as days_since_decision
            ORDER BY blocked_decisions DESC, days_since_decision DESC
            """
            
            params = {"organization_id": organization_id}
            
            async with self.neo4j_service.driver.session() as session:
                result = await session.run(query, params)
                records = await result.data()
                
                bottlenecks = []
                for record in records:
                    decision = dict(record["decision"])
                    bottlenecks.append({
                        "decision": decision,
                        "topics": record["topics"],
                        "blocked_decisions": record["blocked_decisions"],
                        "days_stalled": record["days_since_decision"],
                        "bottleneck_severity": self._calculate_bottleneck_severity(
                            record["blocked_decisions"], 
                            record["days_since_decision"]
                        )
                    })
                
                return bottlenecks
            
        except Exception as e:
            logger.error(f"Failed to find decision bottlenecks: {e}")
            return []
    
    # Analysis Helper Methods
    def _analyze_direct_outcomes(self, outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze direct outcomes of a decision"""
        if not outcomes:
            return {"status": "no_outcomes", "count": 0}
        
        success_scores = [o.get("success_score", 0) for o in outcomes if o.get("success_score") is not None]
        avg_success = sum(success_scores) / len(success_scores) if success_scores else 0
        
        return {
            "total_outcomes": len(outcomes),
            "average_success_score": avg_success,
            "outcome_types": [o.get("outcome_type", "unknown") for o in outcomes],
            "positive_outcomes": len([o for o in outcomes if o.get("success_score", 0) >= 0.7]),
            "negative_outcomes": len([o for o in outcomes if o.get("success_score", 0) < 0.3])
        }
    
    def _analyze_cascade_effects(self, influenced_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cascade effects on future decisions"""
        if not influenced_decisions:
            return {"cascade_count": 0, "impact": "none"}
        
        return {
            "cascade_count": len(influenced_decisions),
            "decision_types_affected": list(set(d.get("decision_type", "") for d in influenced_decisions)),
            "average_success": sum(d.get("success_rating", 0) for d in influenced_decisions) / len(influenced_decisions),
            "impact_level": "high" if len(influenced_decisions) > 5 else "medium" if len(influenced_decisions) > 2 else "low"
        }
    
    def _analyze_topic_impact(self, affected_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze impact on different topics"""
        if not affected_topics:
            return {"topics_affected": 0}
        
        return {
            "topics_affected": len(affected_topics),
            "topic_categories": [t.get("category", "general") for t in affected_topics],
            "high_impact_topics": [t for t in affected_topics if t.get("frequency", 0) > 5]
        }
    
    def _calculate_success_metrics(
        self, 
        decision: Dict[str, Any], 
        outcomes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate various success metrics"""
        success_rating = decision.get("success_rating", 0)
        implementation_progress = decision.get("implementation_progress", 0)
        outcome_success = self._analyze_direct_outcomes(outcomes).get("average_success_score", 0)
        
        return {
            "decision_success_rating": success_rating,
            "implementation_progress": implementation_progress,
            "outcome_success_score": outcome_success,
            "overall_success": (success_rating + implementation_progress/100 + outcome_success) / 3
        }
    
    def _analyze_timeline(
        self, 
        decision: Dict[str, Any], 
        outcomes: List[Dict[str, Any]], 
        influenced_decisions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze timeline and duration aspects"""
        decided_at = decision.get("decided_at")
        if not decided_at:
            return {"timeline": "unknown"}
        
        try:
            decided_date = datetime.fromisoformat(decided_at.replace('Z', '+00:00'))
            days_since = (datetime.utcnow().replace(tzinfo=decided_date.tzinfo) - decided_date).days
            
            return {
                "days_since_decision": days_since,
                "implementation_duration": decision.get("timeline", "unknown"),
                "outcome_timeline": [o.get("occurred_at") for o in outcomes if o.get("occurred_at")],
                "cascade_timeline": len(influenced_decisions)  # Number of subsequent decisions
            }
        except Exception:
            return {"timeline": "parse_error"}
    
    def _identify_risk_factors(
        self, 
        decision: Dict[str, Any], 
        outcomes: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify risk factors based on decision and outcomes"""
        risks = []
        
        # Check implementation progress
        progress = decision.get("implementation_progress", 0)
        if progress < 50:
            risks.append("slow_implementation")
        
        # Check confidence score
        confidence = decision.get("confidence_score", 0)
        if confidence < 0.7:
            risks.append("low_confidence")
        
        # Check outcomes
        negative_outcomes = len([o for o in outcomes if o.get("success_score", 0) < 0.3])
        if negative_outcomes > 0:
            risks.append("negative_outcomes")
        
        # Check urgency vs progress
        urgency = decision.get("urgency", "medium")
        if urgency == "high" and progress < 75:
            risks.append("urgent_but_slow")
        
        return risks
    
    def _extract_lessons_learned(self, outcomes: List[Dict[str, Any]]) -> List[str]:
        """Extract lessons learned from outcomes"""
        lessons = []
        for outcome in outcomes:
            lesson = outcome.get("lessons_learned", "")
            if lesson and lesson not in lessons:
                lessons.append(lesson)
        return lessons
    
    def _calculate_average_success(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate average success rate from patterns"""
        success_rates = [p.get("average_success", 0) for p in patterns if p.get("average_success")]
        return sum(success_rates) / len(success_rates) if success_rates else 0.0
    
    def _extract_common_topics(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Extract most common topics from patterns"""
        all_topics = []
        for pattern in patterns:
            topics = pattern.get("topics", [])
            all_topics.extend(topics)
        
        # Count frequency and return top topics
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)[:10]
    
    def _identify_success_factors(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Identify factors that contribute to success"""
        # This would be more sophisticated in a real implementation
        high_success_patterns = [p for p in patterns if p.get("average_success", 0) > 0.8]
        
        factors = []
        if len(high_success_patterns) > len(patterns) * 0.3:
            factors.append("strong_track_record")
        
        return factors
    
    def _identify_failure_patterns(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns that lead to failure"""
        low_success_patterns = [p for p in patterns if p.get("average_success", 0) < 0.3]
        
        patterns_identified = []
        if len(low_success_patterns) > len(patterns) * 0.2:
            patterns_identified.append("recurring_failures")
        
        return patterns_identified
    
    def _generate_organizational_insights(
        self, 
        patterns: Dict[str, Any], 
        topic_trends: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate insights about organizational decision-making"""
        insights = []
        
        # Analyze decision type performance
        best_type = max(patterns.keys(), key=lambda x: patterns[x]["average_success"])
        worst_type = min(patterns.keys(), key=lambda x: patterns[x]["average_success"])
        
        insights.append(f"Strongest decision-making in {best_type} decisions")
        if patterns[worst_type]["average_success"] < 0.5:
            insights.append(f"Improvement needed in {worst_type} decisions")
        
        # Analyze topic trends
        if topic_trends:
            top_topic = topic_trends[0]["topic"]
            insights.append(f"Most discussed topic: {top_topic}")
        
        return insights
    
    def _generate_recommendations(
        self, 
        patterns: Dict[str, Any], 
        success_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Find decision types that need improvement
        for decision_type, data in patterns.items():
            if data["average_success"] < 0.6:
                recommendations.append(f"Focus on improving {decision_type} decision processes")
        
        return recommendations
    
    def _calculate_topic_similarity(
        self, 
        current_topics: List[str], 
        historical_patterns: List[Dict[str, Any]]
    ) -> float:
        """Calculate similarity between current and historical topics"""
        if not current_topics or not historical_patterns:
            return 0.0
        
        matches = 0
        total_comparisons = 0
        
        for pattern in historical_patterns:
            pattern_topics = pattern.get("topics", [])
            if pattern_topics:
                common_topics = set(current_topics) & set(pattern_topics)
                matches += len(common_topics)
                total_comparisons += max(len(current_topics), len(pattern_topics))
        
        return matches / total_comparisons if total_comparisons > 0 else 0.0
    
    def _calculate_bottleneck_severity(self, blocked_decisions: int, days_stalled: int) -> str:
        """Calculate severity of a decision bottleneck"""
        score = blocked_decisions * 2 + days_stalled / 30
        
        if score > 10:
            return "critical"
        elif score > 5:
            return "high"
        elif score > 2:
            return "medium"
        else:
            return "low"


# Global analyzer instance
decision_flow_analyzer = None


async def get_decision_flow_analyzer() -> DecisionFlowAnalyzer:
    """Get decision flow analyzer instance"""
    global decision_flow_analyzer
    if not decision_flow_analyzer:
        decision_flow_analyzer = DecisionFlowAnalyzer()
        await decision_flow_analyzer.initialize()
    return decision_flow_analyzer
