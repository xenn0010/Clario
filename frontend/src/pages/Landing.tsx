import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import AgentOrbit from '../components/AgentOrbit';
import GlowCard from '../components/GlowCard';
import GraphCanvas from '../components/GraphCanvas';
import { fetchAgents, fetchOrganizationPulse } from '../lib/api';
import type { AgentSummary } from '../types';

const Landing = () => {
  const { data: agents = [], isLoading: loadingAgents } = useQuery({
    queryKey: ['agents', 'demo-enterprise'],
    queryFn: () => fetchAgents('demo-enterprise')
  });

  const { data: pulse } = useQuery({
    queryKey: ['pulse', 'demo-enterprise'],
    queryFn: () => fetchOrganizationPulse('demo-enterprise')
  });

  const showcaseAgents = useMemo<AgentSummary[]>(() => agents.slice(0, 6), [agents]);

  return (
    <div className="pb-24">
      <section className="hero-curve relative overflow-hidden bg-hero-gradient pb-20 pt-24">
        <div className="absolute inset-0 grid-overlay opacity-40" aria-hidden />
        <div className="relative mx-auto grid max-w-6xl gap-16 px-6 md:grid-cols-[1.1fr_0.9fr]">
          <div className="space-y-10">
            <motion.div
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
              className="space-y-6"
            >
              <p className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-sm font-medium uppercase tracking-[0.4em] text-sand/70">
                <span className="h-1.5 w-1.5 rounded-full bg-seafoam" />
                Autonomous Meeting Cloud
              </p>
              <h1 className="max-w-xl font-display text-5xl leading-tight text-sand md:text-6xl">
                Meetings you never have to attend. Decisions you can finally trust.
              </h1>
              <p className="max-w-xl text-lg text-sand/70">
                Clario orchestrates a league of decision-native agents that synthesize intent, surface blind spots, and keep your organization in perfect rhythm around every meeting pulse.
              </p>
              <div className="flex flex-wrap gap-4" id="book-demo">
                <a
                  href="mailto:pilot@clario.ai"
                  className="rounded-full bg-gradient-to-r from-lagoon via-seafoam to-sunburst px-6 py-3 text-sm font-semibold text-slate-950 shadow-floating transition-transform hover:-translate-y-1"
                >
                  Launch a mission
                </a>
                <a
                  href="#experience"
                  className="rounded-full border border-white/20 px-6 py-3 text-sm font-semibold text-sand/80 transition hover:border-seafoam/40 hover:text-sand"
                >
                  Watch the choreography
                </a>
              </div>
            </motion.div>

            <div className="grid gap-6 sm:grid-cols-2">
              <GlowCard title="Resonant memory" eyebrow="Context Loom" accent="seafoam">
                <p>
                  Every agent maintains a living strand of organizational memory, weaving meetings, intents, and knowledge into a single adaptive storyline.
                </p>
                <p className="text-sm text-sand/60">Tailored embeddings · zero drift</p>
              </GlowCard>
              <GlowCard title="Signal-forward" eyebrow="Decision Telemetry" accent="sunburst">
                <p>
                  Predictive attention keeps your teams ahead of bottlenecks, modeling impact scenarios for every outcome before it happens.
                </p>
                <p className="text-sm text-sand/60">Graph-native foresight · flow analytics</p>
              </GlowCard>
            </div>
          </div>

          <div className="flex items-center justify-center">
            {loadingAgents ? (
              <div className="flex h-[420px] w-[420px] items-center justify-center rounded-[48px] border border-white/10 bg-white/5 text-sand/60">
                calibrating agent chorus…
              </div>
            ) : (
              <AgentOrbit agents={showcaseAgents} />
            )}
          </div>
        </div>
        <div className="wave-divider mt-16">
          <svg viewBox="0 0 1440 140" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
            <path
              d="M0,40 C320,120 640,-20 960,30 C1180,70 1300,100 1440,60 L1440,140 L0,140 Z"
              fill="rgba(2, 6, 23, 0.95)"
            />
          </svg>
        </div>
      </section>

      <section id="experience" className="relative mx-auto mt-20 max-w-6xl px-6">
        <div className="grid gap-16 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="space-y-6">
            <p className="text-sm uppercase tracking-[0.4em] text-sand/40">Graph-native intelligence</p>
            <h2 className="font-display text-4xl text-sand">See your organization breathe in real time</h2>
            <p className="text-lg text-sand/70">
              Graph Studio reveals how agents route context, where decision energy concentrates, and how topics ripple through the enterprise. Nodes pulse with live activity and sentiment so you can intervene before friction appears.
            </p>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
                <p className="text-sm uppercase tracking-[0.3em] text-sand/50">Decision velocity</p>
                <p className="mt-3 text-3xl font-semibold text-seafoam">{pulse?.metrics[0]?.value ?? '4.4 days'}</p>
                <p className="mt-2 text-xs text-sand/50">? {pulse?.metrics[0]?.delta ?? 1.2}% faster week over week</p>
              </div>
              <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
                <p className="text-sm uppercase tracking-[0.3em] text-sand/50">Alignment score</p>
                <p className="mt-3 text-3xl font-semibold text-sunburst">{pulse?.metrics[1]?.value ?? '91%'}</p>
                <p className="mt-2 text-xs text-sand/50">Momentum locked across agent mesh</p>
              </div>
            </div>
          </div>
          <div>
            {pulse && <GraphCanvas graph={pulse.graph} />}
          </div>
        </div>
      </section>

      <section className="mx-auto mt-24 max-w-6xl px-6">
        <div className="rounded-[48px] border border-white/10 bg-white/5 p-10 backdrop-blur-xl">
          <div className="grid gap-12 md:grid-cols-3">
            <div>
              <p className="text-sm uppercase tracking-[0.4em] text-sand/40">Agent rituals</p>
              <h3 className="mt-3 font-display text-3xl text-sand">Every meeting crafts a living dossier</h3>
              <p className="mt-4 text-sm text-sand/70">
                Agents co-author postures, commitments, and friction maps so humans receive a cinematic recap instead of a recording.
              </p>
            </div>
            <div className="space-y-6 text-sm text-sand/70">
              <p>
                • Context primes map the emotional and operational posture of every stakeholder.
              </p>
              <p>
                • Decision canvases knit together dependencies from Neo4j so teams can visualize impact arcs.
              </p>
              <p>
                • Resonance loops trigger proactive touchpoints whenever alignment dips.
              </p>
            </div>
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="rounded-[36px] border border-white/10 bg-gradient-to-br from-white/10 via-white/5 to-transparent p-8"
            >
              <p className="text-xs uppercase tracking-[0.4em] text-sand/50">Pilot spotlight</p>
              <p className="mt-4 text-lg text-sand">
                “By the time I wake up, our agents have already negotiated trade-offs and surfaced the three moments that need my human voice. Clario is the first platform that feels alive.”
              </p>
              <p className="mt-6 text-sm font-semibold text-seafoam">Chief of Staff, Series C SaaS</p>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Landing;
