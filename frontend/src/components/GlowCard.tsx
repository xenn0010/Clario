import { motion } from 'framer-motion';
import clsx from 'clsx';

interface GlowCardProps {
  title: string;
  eyebrow?: string;
  accent?: 'lagoon' | 'seafoam' | 'sunburst' | 'coral';
  children: React.ReactNode;
}

const accentMap: Record<NonNullable<GlowCardProps['accent']>, string> = {
  lagoon: 'from-lagoon/30 to-lagoon/10 border-lagoon/40 shadow-[0_20px_60px_-20px_rgba(14,165,233,0.35)]',
  seafoam: 'from-seafoam/30 to-seafoam/10 border-seafoam/40 shadow-[0_20px_60px_-20px_rgba(45,212,191,0.35)]',
  sunburst: 'from-sunburst/30 to-sunburst/10 border-sunburst/40 shadow-[0_20px_60px_-20px_rgba(251,191,36,0.35)]',
  coral: 'from-coral/30 to-coral/10 border-coral/40 shadow-[0_20px_60px_-20px_rgba(249,115,22,0.35)]'
};

const GlowCard = ({ title, eyebrow, accent = 'lagoon', children }: GlowCardProps) => {
  return (
    <motion.article
      whileHover={{ translateY: -8 }}
      className={clsx(
        'group relative overflow-hidden rounded-[32px] border border-white/5 bg-gradient-to-br p-[1px] transition-transform',
        accentMap[accent]
      )}
    >
      <div className="card-curve h-full rounded-[31px] p-6">
        {eyebrow && <p className="text-sm font-medium uppercase tracking-[0.3em] text-sand/50">{eyebrow}</p>}
        <h3 className="mt-3 font-display text-2xl text-sand">{title}</h3>
        <div className="mt-6 space-y-4 text-sm text-sand/80">{children}</div>
      </div>
      <motion.div
        aria-hidden
        className="pointer-events-none absolute -right-10 top-10 h-40 w-40 rounded-full opacity-20 blur-3xl"
        style={{ background: 'radial-gradient(circle, rgba(255,255,255,0.8), transparent 60%)' }}
        animate={{ scale: [1, 1.2, 1], opacity: [0.2, 0.35, 0.2] }}
        transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
      />
    </motion.article>
  );
};

export default GlowCard;
