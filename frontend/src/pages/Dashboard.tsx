import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { fetchActiveMeetings, fetchAgents } from '../lib/api';
import type { AgentInteraction, AgentSummary } from '../types';

const sentimentAccent: Record<AgentInteraction['sentiment'], string> = {
  optimistic: 'text-seafoam',
  cautious: 'text-sunburst',
  critical: 'text-coral'
};

const Dashboard = () => {
  const { data: agents = [] } = useQuery({
    queryKey: ['agents', 'dashboard'],
    queryFn: () => fetchAgents('demo-enterprise')
  });

  const { data: meetings } = useQuery({
    queryKey: ['meetings', 'active'],
    queryFn: fetchActiveMeetings
  });

  const interactions = useMemo<AgentInteraction[]>(
    () => [
      {
        author: 'Sora Liang',
        sentiment: 'optimistic',
        content: 'Synthesized 14 market signals—momentum favors an accelerated launch if we unlock ops capacity.',
        timestamp: '5 min ago',
        action: 'Proposes: allocate +8% headcount to automation pod'
      },
      {
        author: 'Kai Moreno',
        sentiment: 'cautious',
        content: 'Ops automation unlocks capacity but raises risk of service latency during migration. Suggest phased rollout.',
        timestamp: '3 min ago',
        action: 'Mitigation: dual-track 3-week shadow period'
      },
      {
        author: 'Vela Anwar',
        sentiment: 'optimistic',
        content: 'Customer cohorts respond well to expedited roadmap so long as communications cadence doubles in week one.',
        timestamp: '1 min ago',
        action: 'Triggered: CX clarity strand'
      }
    ],
    []
  );

  return (
    <div className="mx-auto max-w-6xl space-y-16 px-6 py-16">
      <header className="space-y-4">
        <p className="text-sm uppercase tracking-[0.4em] text-sand/40">Agent operations cockpit</p>
        <h1 className="font-display text-4xl text-sand">Command the chorus</h1>
        <p className="max-w-2xl text-sand/70">
          Observe how agents negotiate strategy, keep meetings humming, and stabilize decision flow. Everything updates live—humans simply fine-tune the score.
        </p>
      </header>

      <section className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-2xl text-sand">Agent council</h2>
            <span className="rounded-full bg-white/5 px-4 py-2 text-xs uppercase tracking-[0.3em] text-sand/60">
              {agents.length} active
            </span>
          </div>
          <div className="rounded-[40px] border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
            <div className="grid gap-4 md:grid-cols-2">
              {agents.map((agent) => (
                <motion.div
                  key={agent.agent_id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4 }}
                  className="flex flex-col rounded-3xl border border-white/10 bg-slate-950/70 p-5"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-[0.3em] text-sand/50">{agent.role}</p>
                      <p className="mt-2 text-lg font-semibold text-sand">{agent.name}</p>
                    </div>
                    <span className="mt-1 inline-flex items-center gap-2 rounded-full bg-white/5 px-3 py-1 text-xs text-sand/70">
                      <span className={`h-2.5 w-2.5 rounded-full ${agent.status === 'online' ? 'bg-seafoam' : 'bg-sunburst'}`} />
                      {agent.status}
                    </span>
                  </div>
                  <p className="mt-4 text-sm text-sand/60">{agent.personality}</p>
                  <div className="mt-4 flex flex-wrap gap-2 text-xs text-sand/60">
                    {agent.expertise_areas.slice(0, 3).map((area) => (
                      <span key={area} className="rounded-full bg-white/5 px-3 py-1">
                        {area}
                      </span>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        <div className="rounded-[40px] border border-white/10 bg-gradient-to-br from-white/10 to-transparent p-8 backdrop-blur-xl">
          <h2 className="font-display text-2xl text-sand">Live agent discourse</h2>
          <p className="mt-3 text-sm text-sand/60">
            Every exchange is tracked, weighted, and woven into the graph. Tap into their current huddle.
          </p>
          <div className="mt-6 space-y-4">
            {interactions.map((interaction, index) => (
              <motion.article
                key={interaction.author}
                initial={{ opacity: 0, x: 24 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="rounded-3xl border border-white/10 bg-slate-950/70 p-5"
              >
                <div className="flex items-center justify-between text-xs text-sand/50">
                  <span>{interaction.timestamp}</span>
                  <span className={sentimentAccent[interaction.sentiment]}>{interaction.sentiment}</span>
                </div>
                <p className="mt-3 text-sm text-sand">{interaction.content}</p>
                {interaction.action && <p className="mt-3 text-xs text-seafoam">{interaction.action}</p>}
              </motion.article>
            ))}
          </div>
        </div>
      </section>

      <section className="grid gap-10 md:grid-cols-[0.8fr_1.2fr]">
        <div className="rounded-[40px] border border-white/10 bg-white/5 p-8 backdrop-blur-xl">
          <h2 className="font-display text-2xl text-sand">Meeting cadence</h2>
          <p className="mt-2 text-sm text-sand/60">Agents keep every ritual on tempo.</p>
          <div className="mt-6 space-y-4">
            {meetings?.active_meetings.map((meeting) => (
              <motion.div
                key={meeting.meeting_id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
                className="rounded-3xl border border-white/10 bg-slate-950/70 p-5"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs uppercase tracking-[0.3em] text-sand/50">{meeting.status}</p>
                    <p className="mt-2 font-semibold text-sand">{meeting.title}</p>
                  </div>
                  <span className="rounded-full bg-white/5 px-3 py-1 text-xs text-sand/60">
                    {meeting.participant_agents.length} agents
                  </span>
                </div>
                <p className="mt-4 text-xs text-sand/60">
                  {meeting.participant_agents.join(' • ')}
                </p>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="rounded-[40px] border border-white/10 bg-gradient-to-br from-lagoon/20 via-slate-950 to-slate-950 p-8 backdrop-blur-xl text-sand">
          <h2 className="font-display text-2xl">Decision resonance</h2>
          <p className="mt-3 text-sm text-sand/70">
            Graph DB simulations show the downstream impact of agent choices. Drag the timeline to re-simulate potential futures.
          </p>
          <div className="mt-6 grid gap-4 text-sm text-sand/70">
            <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5">
              <p className="text-xs uppercase tracking-[0.3em] text-sand/50">Scenario: Accelerated launch</p>
              <p className="mt-3 text-sand">
                Predicted conversion lift <span className="text-seafoam">+12%</span> if ops friction <span className="text-sunburst">-18%</span> by sprint 2.
              </p>
            </div>
            <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5">
              <p className="text-xs uppercase tracking-[0.3em] text-sand/50">Scenario: Maintain cadence</p>
              <p className="mt-3 text-sand">
                Stabilized churn at <span className="text-seafoam">-3.4%</span>; opportunity cost <span className="text-coral">-9%</span> projected ARR growth.
              </p>
            </div>
          </div>
          <motion.div
            className="mt-8 h-2 rounded-full bg-white/10"
            initial={{ backgroundPositionX: '0%' }}
            animate={{ backgroundPositionX: '200%' }}
            transition={{ duration: 14, repeat: Infinity, ease: 'linear' }}
            style={{
              backgroundImage: 'linear-gradient(90deg, rgba(14,165,233,0.6), rgba(45,212,191,0.6), rgba(251,191,36,0.6), rgba(14,165,233,0.6))'
            }}
          />
        </div>
      </section>
    </div>
  );
};

export default Dashboard;
