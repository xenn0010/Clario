import { motion } from 'framer-motion';
import type { AgentSummary } from '../types';

interface AgentOrbitProps {
  agents: AgentSummary[];
}

const ringPositions = [
  { x: 0, y: -140 },
  { x: 130, y: -60 },
  { x: 150, y: 100 },
  { x: 0, y: 160 },
  { x: -150, y: 100 },
  { x: -140, y: -70 }
];

const AgentOrbit = ({ agents }: AgentOrbitProps) => {
  return (
    <div className="relative h-[420px] w-[420px]">
      <motion.div
        className="absolute inset-6 rounded-full border border-dashed border-seafoam/40"
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 60, ease: 'linear' }}
      />
      <motion.div
        className="absolute inset-20 rounded-full border border-dashed border-lagoon/30"
        animate={{ rotate: -360 }}
        transition={{ repeat: Infinity, duration: 80, ease: 'linear' }}
      />
      <div className="absolute inset-0 flex items-center justify-center">
        <motion.div
          className="flex h-40 w-40 items-center justify-center rounded-full bg-gradient-to-br from-lagoon/30 to-seafoam/20 text-center text-sand"
          animate={{ boxShadow: ['0 0 0 rgba(14,165,233,0.2)', '0 0 45px rgba(45,212,191,0.35)', '0 0 0 rgba(14,165,233,0.2)'] }}
          transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
        >
          <div>
            <p className="text-xs uppercase tracking-[0.4em] text-sand/60">Clario</p>
            <p className="mt-1 font-display text-xl">Agent Chorus</p>
          </div>
        </motion.div>
      </div>
      {agents.slice(0, ringPositions.length).map((agent, index) => (
        <motion.div
          key={agent.agent_id}
          className="absolute flex w-48 flex-col items-center rounded-3xl border border-white/10 bg-slate-950/80 p-4 backdrop-blur-xl"
          style={{
            top: `calc(50% + ${ringPositions[index].y}px - 4rem)`,
            left: `calc(50% + ${ringPositions[index].x}px - 4rem)`
          }}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: index * 0.1 + 0.3 }}
        >
          <span className="text-xs uppercase tracking-[0.3em] text-sand/50">{agent.role}</span>
          <p className="mt-1 font-semibold text-sand">{agent.name}</p>
          <p className="mt-2 text-xs text-sand/60">{agent.expertise_areas.slice(0, 2).join(' • ')}</p>
          <motion.span
            className="mt-3 inline-flex items-center gap-2 rounded-full bg-white/5 px-3 py-1 text-xs"
            animate={{ opacity: [0.6, 1, 0.6] }}
            transition={{ duration: 4 + index, repeat: Infinity }}
          >
            <span className={`h-2 w-2 rounded-full ${agent.status === 'online' ? 'bg-seafoam' : 'bg-sunburst'}`} />
            {agent.personality}
          </motion.span>
        </motion.div>
      ))}
    </div>
  );
};

export default AgentOrbit;
