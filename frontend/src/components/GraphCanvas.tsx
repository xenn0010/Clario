import { useEffect, useMemo, useState } from 'react';
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation } from 'd3-force';
import { motion, AnimatePresence } from 'framer-motion';
import type { GraphData, GraphNode, GraphLink } from '../types';

interface LayoutNode extends GraphNode {
  x: number;
  y: number;
  vx?: number;
  vy?: number;
}

interface GraphCanvasProps {
  graph: GraphData;
  focusNodes?: string[];
}

const NODE_COLORS: Record<GraphNode['type'], string> = {
  agent: '#2DD4BF',
  meeting: '#0EA5E9',
  decision: '#FBBF24',
  topic: '#38BDF8',
  outcome: '#34D399'
};

const GraphCanvas = ({ graph, focusNodes = [] }: GraphCanvasProps) => {
  const [nodes, setNodes] = useState<LayoutNode[]>(() =>
    graph.nodes.map((node, index) => ({
      ...node,
      x: Math.random() * 720 + 100,
      y: Math.random() * 420 + 80,
      vx: 0,
      vy: 0,
      activity: node.activity ?? ((index % 5) + 1) / 5
    }))
  );
  const [links, setLinks] = useState<GraphLink[]>(graph.links);
  const [hoverNode, setHoverNode] = useState<LayoutNode | null>(null);

  useEffect(() => {
    const layoutNodes: LayoutNode[] = graph.nodes.map((node, index) => ({
      ...node,
      x: Math.random() * 720 + 100,
      y: Math.random() * 420 + 80,
      vx: 0,
      vy: 0,
      activity: node.activity ?? ((index % 5) + 1) / 5
    }));
    const layoutLinks = graph.links.map((link) => ({ ...link }));

    const simulation = forceSimulation(layoutNodes)
      .force(
        'link',
        forceLink(layoutLinks)
          .id((d: any) => d.id)
          .distance((d: any) => 220 - ((d.intensity ?? 0.5) * 120))
          .strength(0.6)
      )
      .force('charge', forceManyBody().strength(-220))
      .force('center', forceCenter(480, 280))
      .force('collide', forceCollide().radius(60));

    simulation.on('tick', () => {
      setNodes(layoutNodes.map((node) => ({ ...node })));
      setLinks(layoutLinks.map((link) => ({ ...link })));
    });

    return () => {
      simulation.stop();
    };
  }, [graph]);

  const focusSet = useMemo(() => new Set(focusNodes), [focusNodes]);
  const hasFocus = focusSet.size > 0;

  const linkElements = useMemo(() => {
    return links.map((link, index) => {
      const source = nodes.find((node) => node.id === (link.source as string));
      const target = nodes.find((node) => node.id === (link.target as string));

      if (!source || !target) {
        return null;
      }

      const intensity = link.intensity ?? 0.4;
      const highlighted = !hasFocus || focusSet.has(source.id) || focusSet.has(target.id);

      return (
        <line
          key={`${link.source}-${link.target}-${index}`}
          x1={source.x}
          y1={source.y}
          x2={target.x}
          y2={target.y}
          stroke={`rgba(226, 232, 240, ${highlighted ? 0.16 + intensity * 0.25 : 0.05})`}
          strokeWidth={highlighted ? 1.4 + intensity * 1.6 : 0.8}
          strokeLinecap="round"
        />
      );
    });
  }, [links, nodes, focusSet, hasFocus]);

  return (
    <div className="relative h-[560px] w-full overflow-hidden rounded-[40px] border border-white/5 bg-white/5 backdrop-blur-xl">
      <svg viewBox="0 0 960 560" className="h-full w-full">
        <defs>
          <radialGradient id="glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(226,232,240,0.35)" />
            <stop offset="100%" stopColor="rgba(226,232,240,0)" />
          </radialGradient>
        </defs>

        <rect width="960" height="560" fill="url(#glow)" opacity={0.08} />
        <g>{linkElements}</g>
        <g>
          {nodes.map((node) => {
            const highlighted = !hasFocus || focusSet.has(node.id);
            const baseRadius = 24 + (node.activity ?? 0.4) * 18;
            return (
              <motion.g
                key={node.id}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: highlighted ? 1 : 0.95, opacity: highlighted ? 1 : 0.35 }}
                transition={{ duration: 0.6, delay: Math.random() * 0.3 }}
                onMouseEnter={() => setHoverNode(node)}
                onMouseLeave={() => setHoverNode((prev) => (prev?.id === node.id ? null : prev))}
              >
                <motion.circle
                  cx={node.x}
                  cy={node.y}
                  r={baseRadius}
                  fill={`rgba(15, 23, 42, 0.9)`}
                  stroke={NODE_COLORS[node.type]}
                  strokeWidth={highlighted ? 3 : 1.5}
                  animate={{
                    filter: highlighted
                      ? [`drop-shadow(0px 0px 0px ${NODE_COLORS[node.type]})`, `drop-shadow(0px 0px 18px ${NODE_COLORS[node.type]}40)`]
                      : 'drop-shadow(0px 0px 0px rgba(0,0,0,0))'
                  }}
                  transition={{ repeat: highlighted ? Infinity : 0, repeatType: 'mirror', duration: 4, ease: 'easeInOut' }}
                />
                <text
                  x={node.x}
                  y={node.y + 5}
                  textAnchor="middle"
                  className="fill-sand text-sm font-semibold"
                  opacity={highlighted ? 1 : 0.4}
                >
                  {node.label}
                </text>
              </motion.g>
            );
          })}
        </g>
      </svg>
      <AnimatePresence>
        {hoverNode && (
          <motion.div
            key={hoverNode.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="pointer-events-none absolute bottom-6 left-1/2 w-[340px] -translate-x-1/2 rounded-3xl border border-white/10 bg-slate-950/85 p-5 text-sand shadow-floating"
          >
            <p className="text-xs uppercase tracking-[0.25em] text-sand/50">{hoverNode.type}</p>
            <p className="mt-2 font-display text-xl">{hoverNode.label}</p>
            <p className="mt-2 text-sm text-sand/70">
              Activity index: {(hoverNode.activity ?? 0.45).toFixed(2)}
            </p>
            {hoverNode.sentiment && (
              <p className="mt-2 text-xs text-sand/60">Sentiment: {hoverNode.sentiment}</p>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default GraphCanvas;
