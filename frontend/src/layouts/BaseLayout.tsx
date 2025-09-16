import { useState } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import AnimatedBackdrop from '../components/AnimatedBackdrop';

const navLinks = [
  { to: '/', label: 'Experience' },
  { to: '/dashboard', label: 'Agent Ops' },
  { to: '/graph-studio', label: 'Graph Studio' }
];

interface BaseLayoutProps {
  children: React.ReactNode;
}

const BaseLayout = ({ children }: BaseLayoutProps) => {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-950">
      <AnimatedBackdrop />
      <div className="relative z-10 flex min-h-screen flex-col">
        <header className="sticky top-0 z-50 bg-slate-950/60 backdrop-blur-xl">
          <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
            <Link to="/" className="flex items-center gap-3 text-sand">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-lagoon to-seafoam text-lg font-semibold text-slate-950 shadow-floating">
                CA
              </span>
              <div>
                <p className="font-display text-xl tracking-wide">Clario Autonomous</p>
                <p className="text-xs uppercase tracking-[0.3em] text-sand/60">Meetings reimagined</p>
              </div>
            </Link>
            <nav className="hidden items-center gap-2 md:flex">
              {navLinks.map((link) => (
                <NavLink
                  key={link.to}
                  to={link.to}
                  className={({ isActive }) =>
                    `rounded-full px-5 py-2 text-sm font-medium transition-all duration-300 ${
                      isActive
                        ? 'bg-white/10 text-sunburst shadow-floating'
                        : 'text-sand/70 hover:bg-white/5 hover:text-sand'
                    }`
                  }
                >
                  {link.label}
                </NavLink>
              ))}
              <a
                href="#book-demo"
                className="rounded-full bg-gradient-to-r from-lagoon via-seafoam to-sunburst px-5 py-2 text-sm font-semibold text-slate-950 shadow-floating transition-transform hover:-translate-y-0.5"
              >
                Book a holo-briefing
              </a>
            </nav>
            <button
              className="relative z-50 flex h-11 w-11 items-center justify-center rounded-full border border-white/10 text-sand md:hidden"
              onClick={() => setMenuOpen((prev) => !prev)}
              aria-label="Toggle menu"
            >
              <span className="sr-only">Toggle menu</span>
              <motion.span
                className="block h-px w-5 bg-current"
                animate={{ rotate: menuOpen ? 45 : 0, y: menuOpen ? 1 : -3 }}
                transition={{ duration: 0.3 }}
              />
              <motion.span
                className="block h-px w-5 bg-current"
                animate={{ opacity: menuOpen ? 0 : 1 }}
                transition={{ duration: 0.3 }}
              />
              <motion.span
                className="block h-px w-5 bg-current"
                animate={{ rotate: menuOpen ? -45 : 0, y: menuOpen ? -1 : 3 }}
                transition={{ duration: 0.3 }}
              />
            </button>
          </div>
          <AnimatePresence>
            {menuOpen && (
              <motion.nav
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="md:hidden"
              >
                <div className="space-y-2 px-6 pb-6">
                  {navLinks.map((link) => (
                    <NavLink
                      key={link.to}
                      to={link.to}
                      onClick={() => setMenuOpen(false)}
                      className={({ isActive }) =>
                        `block rounded-2xl bg-white/5 px-5 py-4 text-lg font-semibold tracking-wide transition-colors ${
                          isActive ? 'text-sunburst' : 'text-sand/80 hover:text-sand'
                        }`
                      }
                    >
                      {link.label}
                    </NavLink>
                  ))}
                  <a
                    href="#book-demo"
                    className="block rounded-2xl bg-gradient-to-r from-lagoon via-seafoam to-sunburst px-5 py-4 text-center text-lg font-semibold text-slate-950"
                  >
                    Book a holo-briefing
                  </a>
                </div>
              </motion.nav>
            )}
          </AnimatePresence>
        </header>
        <main className="relative flex-1">{children}</main>
        <footer className="border-t border-white/5 bg-slate-950/70">
          <div className="mx-auto flex max-w-6xl flex-col gap-6 px-6 py-10 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="font-display text-lg text-sand">A signature Clario experience</p>
              <p className="text-sm text-sand/60">Crafted for autonomous teams navigating complex decisions.</p>
            </div>
            <div className="flex gap-4 text-sm text-sand/60">
              <a href="mailto:hello@clario.ai" className="hover:text-seafoam">
                hello@clario.ai
              </a>
              <a href="https://clario.ai" className="hover:text-seafoam">
                clario.ai
              </a>
              <span>© {new Date().getFullYear()} Clario</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default BaseLayout;
