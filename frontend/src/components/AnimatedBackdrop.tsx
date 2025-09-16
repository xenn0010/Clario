import { motion } from 'framer-motion';

const AnimatedBackdrop = () => {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <motion.div
        className="glow-dot teal w-[320px] h-[320px] -top-10 -left-10"
        animate={{
          x: [0, 40, -20, 0],
          y: [0, 20, -30, 0],
          opacity: [0.65, 0.75, 0.6, 0.65]
        }}
        transition={{ duration: 18, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="glow-dot blue w-[260px] h-[260px] -bottom-16 right-12"
        animate={{
          x: [0, -30, 10, 0],
          y: [0, -15, 25, 0],
          opacity: [0.6, 0.8, 0.55, 0.6]
        }}
        transition={{ duration: 22, repeat: Infinity, ease: 'easeInOut', delay: 3 }}
      />
      <motion.div
        className="glow-dot gold w-[220px] h-[220px] top-28 right-[35%]"
        animate={{
          x: [0, 25, -15, 0],
          y: [0, 30, 10, 0],
          opacity: [0.55, 0.7, 0.5, 0.55]
        }}
        transition={{ duration: 16, repeat: Infinity, ease: 'easeInOut', delay: 6 }}
      />
    </div>
  );
};

export default AnimatedBackdrop;
