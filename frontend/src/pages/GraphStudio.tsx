import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import GraphCanvas from '../components/GraphCanvas';
import { fetchOrganizationPulse } from '../lib/api';

const scenarios = [
  {
    id: 'optimistic',
    name: 'Momentum surge',
    description: 'Runway to accelerate product launch by 3 weeks while preserving service reliability.',
    highlight: ['agent-sora', 'decision-supply']
  },
  {
    id: 'balanced',
    name: 'Stability orbit',
    description: 'Optimize for organizational alignment and minimize burn across supporting squads.',
    highlight: ['agent-ayan', 'meeting-aurora']
  },
  {
    id: 'rescue',
    name: 'Bottleneck rescue',
    description: 'Target redline nodes creating friction and reroute knowledge through resilient agents.',
    highlight: ['topic-latency']
  }
];

const GraphStudio = () => {
  const [scenario, setScenario] = useState(scenarios[0]);
  const { data: pulse, isLoading, isFetching } = useQuery({
    queryKey: ['pulse', 'graph-studio'],
    queryFn: () => fetchOrganizationPulse('demo-enterprise')
  });

  const metrics = useMemo(() => pulse?.metrics ?? [], [pulse]);

  return (
    <div className="mx-auto max-w-6xl space-y-16 px-6 py-16">
      <header className="space-y-5">
        <p className="text-sm uppercase tracking-[0.4em] text-sand/40">Graph studio</p>
        <h1 className="font-display text-4xl text-sand">Simulate the living decision graph</h1>
        <p className="max-w-2xl text-sand/70">
          Neo4j-backed simulations allow you to explore cascading effects. Pick a scenario and watch the constellation adapt.
        </p>
      </header>

      <section className="grid gap-12 lg:grid-cols-[1.2fr_0.8fr]">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-2xl text-sand">Real-time constellation</h2>
            {isFetching && <span className="text-xs text-sand/50">updating telemetry…</span>}
          </div>
          {isLoading || !pulse ? (
            <div className="flex h-[560px] items-center justify-center rounded-[40px] border border-white/10 bg-white/5 text-sand/60">
              generating graph fabric…
            </div>
          ) : (
            <GraphCanvas graph={pulse.graph} focusNodes={scenario.highlight} />
          )}
        </div>

        <div className="space-y-8">
          <div className="rounded-[36px] border border-white/10 bg-white/5 p-6">
            <p className="text-xs uppercase tracking-[0.3em] text-sand/50">Simulation scenarios</p>
            <div className="mt-4 space-y-3">
              {scenarios.map((item) => (
                <motion.button
                  key={item.id}
                  onClick={() => setScenario(item)}
                  whileHover={{ scale: 1.01 }}
                  className={`w-full rounded-3xl border px-5 py-4 text-left transition ${
                    scenario.id === item.id
                      ? 'border-seafoam/60 bg-slate-950/70 text-sand'
                      : 'border-white/10 bg-slate-950/40 text-sand/70 hover:text-sand'
                  }`}
                >
                  <p className="text-sm font-semibold">{item.name}</p>
                  <p className="mt-2 text-xs text-sand/60">{item.description}</p>
                </motion.button>
              ))}
            </div>
          </div>
          <div className="rounded-[36px] border border-white/10 bg-gradient-to-br from-slate-900/80 via-slate-950 to-slate-950 p-6">
            <p className="text-xs uppercase tracking-[0.3em] text-sand/40">Graph telemetry</p>
            <div className="mt-4 space-y-5">
              {metrics.map((metric) => (
                <div key={metric.name} className="rounded-3xl border border-white/10 bg-slate-950/70 p-5">
                  <p className="text-xs uppercase tracking-[0.3em] text-sand/50">{metric.name}</p>
                  <p className="mt-3 text-2xl font-semibold text-sand">{metric.value}</p>
                  {metric.delta !== undefined && (
                    <p className={`mt-2 text-xs ${metric.delta >= 0 ? 'text-seafoam' : 'text-coral'}`}>
                      {metric.delta >= 0 ? '+' : ''}
                      {metric.delta}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
          <div className="rounded-[36px] border border-white/10 bg-white/5 p-6">
            <p className="text-xs uppercase tracking-[0.3em] text-sand/40">Scenario focus</p>
            <p className="mt-3 text-sm text-sand/70">
              Highlighting nodes: <span className="text-seafoam">{scenario.highlight.join(', ')}</span>
            </p>
            <p className="mt-4 text-xs text-sand/50">
              Toggle scenarios to explore how the graph reorganizes around decision energy. Graph animations are powered by d3-force on top of Clario&apos;s Neo4j projections.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default GraphStudio;
