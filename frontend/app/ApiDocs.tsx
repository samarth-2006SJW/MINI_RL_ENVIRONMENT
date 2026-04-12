import { Code2, Zap, RefreshCw, ArrowRight } from "lucide-react";

const endpoints = [
  {
    method: "POST",
    path: "/reset",
    desc: "Reset environment to initial state. Returns first observation.",
    body: `{ "scenario": "easy" | "medium" | "hard" }`,
    color: "text-sky-400 bg-sky-500/10 border-sky-500/30",
  },
  {
    method: "POST",
    path: "/step",
    desc: "Execute one agent action. Returns (obs, reward, terminated, info).",
    body: `{ "action_type": "NAVIGATE", "target_robot_id": "R1", ... }`,
    color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
  },
  {
    method: "GET",
    path: "/state",
    desc: "Get the current warehouse state without advancing a step.",
    body: null,
    color: "text-violet-400 bg-violet-500/10 border-violet-500/30",
  },
  {
    method: "POST",
    path: "/api/simulate/stream",
    desc: "Stream a full episode over SSE. Each event is a JSON frame.",
    body: `{ "scenario": "medium", "max_steps": 20 }`,
    color: "text-amber-400 bg-amber-500/10 border-amber-500/30",
  },
];

const ApiDocs = () => {
  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border/50 bg-secondary/50 text-xs text-muted-foreground mb-4">
          <Code2 className="w-3 h-3" />
          OpenEnv API
        </div>
        <h2 className="text-2xl font-heading font-bold text-foreground">API Reference</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Standard OpenEnv endpoints — compatible with any PPO / DQN agent that uses the reset/step interface.
        </p>
      </div>

      {/* Base URL */}
      <div className="glass-card p-4 rounded-xl flex items-center gap-3">
        <Zap className="w-4 h-4 text-primary shrink-0" />
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">Base URL</p>
          <code className="text-sm text-foreground font-mono">http://localhost:7860</code>
        </div>
      </div>

      {/* Endpoints */}
      <div className="space-y-4">
        {endpoints.map((ep) => (
          <div key={ep.path} className="glass-card p-5 rounded-xl space-y-3">
            <div className="flex items-center gap-3">
              <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded border ${ep.color}`}>
                {ep.method}
              </span>
              <code className="text-sm text-foreground font-mono">{ep.path}</code>
            </div>
            <p className="text-sm text-muted-foreground">{ep.desc}</p>
            {ep.body && (
              <div className="bg-black/40 rounded-lg p-3 border border-white/5">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Request Body</p>
                <code className="text-xs text-emerald-400 font-mono break-all">{ep.body}</code>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Action Types */}
      <div className="glass-card p-5 rounded-xl">
        <div className="flex items-center gap-2 mb-4">
          <RefreshCw className="w-4 h-4 text-primary" />
          <h3 className="font-heading font-semibold text-sm">Supported Action Types</h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {["NAVIGATE", "RESTOCK_INVENTORY", "DISPATCH_MAINTENANCE", "REROUTE_PATH", "RESOLVE_EXCEPTION", "NO_OP"].map((a) => (
            <div key={a} className="flex items-center gap-2 text-xs">
              <ArrowRight className="w-3 h-3 text-primary shrink-0" />
              <code className="text-foreground font-mono">{a}</code>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ApiDocs;
