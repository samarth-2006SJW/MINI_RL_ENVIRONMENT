import React, { useState, useEffect, useRef } from "react";
import { Play, Square, Activity, Target, AlertCircle, Terminal } from "lucide-react";
import { Button } from "@/components/ui/button";
import WarehouseGrid from "./WarehouseGrid";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const SimulationDashboard = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [scenario, setScenario] = useState("medium");
  const [maxSteps, setMaxSteps] = useState(20);
  
  const [gridState, setGridState] = useState(null);
  const [plotData, setPlotData] = useState([]);
  const [actionLog, setActionLog] = useState<string[]>([]);
  const [kpis, setKpis] = useState({
    reward: 0,
    steps: 0,
    efficiency: 0,
    exceptions: 0
  });

  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [actionLog]);

  // Load initial grid geometry
  useEffect(() => {
    if (isRunning) return;
    
    fetch(`/api/scenario/${scenario.trim()}`)
      .then(res => res.json())
      .then(data => {
        if (data.grid_state) {
          setGridState(data.grid_state);
        }
      })
      .catch(err => console.error("Failed to fetch initial grid state:", err));
  }, [scenario, isRunning]);

  const startSimulation = async () => {
    if (isRunning) return;
    setIsRunning(true);
    setPlotData([]);
    setActionLog(["[SYS] Initializing UPLINK... Environment resetting."]);

    try {
      const response = await fetch("/api/simulate/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario, max_steps: maxSteps })
      });

      if (!response.body) throw new Error("No readable stream");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        // Process full SSE lines
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || ""; // Keep the last incomplete chunk in buffer
        
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.substring(6);
            try {
              const data = JSON.parse(dataStr);
              setGridState(data.grid_state);
              setPlotData(data.plot_data || []);
              setActionLog(data.action_log || []);
              setKpis({
                reward: data.total_reward || 0,
                steps: data.step || 0,
                efficiency: data.action_efficiency || 0,
                exceptions: data.exceptions_remaining || 0
              });

              if (data.terminated) {
                 setIsRunning(false);
                 setActionLog(prev => [...prev, "[SYS] ✓ Target Reached — All objectives resolved."]);
              }
            } catch (err) {
              console.error("Failed to parse SSE JSON:", err);
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      setActionLog(prev => [...prev, `[ERROR] Connection failed: ${err}`]);
      setIsRunning(false);
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-card p-4 rounded-2xl">
        <div>
          <h2 className="text-2xl font-heading font-bold text-foreground">Live Telemetry</h2>
          <p className="text-sm text-muted-foreground">Monitor real-time agent decisions & environment states</p>
        </div>
        
        <div className="flex items-center gap-3">
          <select 
            className="bg-secondary/50 border border-border/50 text-sm rounded-lg px-3 py-2 text-foreground focus:ring-1 focus:ring-primary outline-none"
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            disabled={isRunning}
          >
            <option value="easy">Op-Easy: Clear Paths</option>
            <option value="medium">Op-Medium: Standard Load</option>
            <option value="hard">Op-Hard: Incident Surge</option>
          </select>
          
          <input 
            type="number" 
            className="w-20 bg-secondary/50 border border-border/50 text-sm rounded-lg px-3 py-2 text-foreground focus:ring-1 focus:ring-primary outline-none"
            value={maxSteps}
            onChange={(e) => setMaxSteps(Number(e.target.value))}
            min={5} max={100}
            disabled={isRunning}
          />

          <Button 
            onClick={startSimulation} 
            disabled={isRunning}
            className={`gap-2 ${isRunning ? 'bg-secondary text-muted-foreground' : 'bg-primary text-primary-foreground hover:bg-primary/90'} shadow-lg`}
          >
            {isRunning ? <Square className="w-4 h-4 animate-pulse" /> : <Play className="w-4 h-4" />}
            {isRunning ? 'Running...' : 'Run Simulation'}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Col: Grid & Plot */}
        <div className="lg:col-span-2 space-y-6">
          <div className="glass-card p-1 rounded-2xl relative">
            <div className="absolute top-4 left-4 z-20 flex items-center gap-2 bg-black/60 px-3 py-1.5 rounded-full border border-white/10 backdrop-blur-md">
              <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-success animate-pulse' : 'bg-muted'}`} />
              <span className="text-[10px] font-mono font-medium text-white/80 uppercase tracking-wider">
                Sector Feed {isRunning ? 'Live' : 'Standby'}
              </span>
            </div>
            <WarehouseGrid gridState={gridState} />
          </div>

          <div className="glass-card p-6 rounded-2xl">
             <div className="flex items-center gap-2 mb-4">
                <Activity className="w-4 h-4 text-primary" />
                <h3 className="font-heading font-semibold text-sm">Reward Trajectory</h3>
             </div>
             <div className="w-full" style={{ minHeight: 0 }}>
               {plotData.length > 0 ? (
                 <ResponsiveContainer width="100%" height={200}>
                   <LineChart data={plotData} margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
                     <XAxis dataKey="step" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                     <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `${value}`} />
                     <Tooltip 
                       contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                       itemStyle={{ color: '#fff' }}
                     />
                     <Line 
                        type="monotone" 
                        dataKey="total_reward" 
                        stroke="hsl(var(--primary))" 
                        strokeWidth={2} 
                        dot={false}
                        activeDot={{ r: 4, fill: "hsl(var(--primary))" }}
                      />
                   </LineChart>
                 </ResponsiveContainer>
               ) : (
                 <div className="h-[200px] flex flex-col items-center justify-center border border-dashed border-border/50 rounded-xl text-muted-foreground text-sm">
                   Run a simulation to see the reward chart
                 </div>
               )}
             </div>
          </div>
        </div>

        {/* Right Col: KPIs & Terminal */}
        <div className="space-y-6 flex flex-col h-full">
          {/* KPIs */}
          <div className="grid grid-cols-2 gap-4">
            <div className="glass-card p-4 rounded-xl">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Total Reward</span>
                <Target className="w-3 h-3 text-primary" />
              </div>
              <p className="text-2xl font-heading font-bold text-foreground">
                {kpis.reward.toFixed(2)}
              </p>
            </div>
            
            <div className="glass-card p-4 rounded-xl">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Efficiency</span>
                <Activity className="w-3 h-3 text-secondary-foreground" />
              </div>
              <p className="text-2xl font-heading font-bold text-foreground">
                {(kpis.efficiency * 100).toFixed(0)}%
              </p>
            </div>

            <div className="glass-card p-4 rounded-xl">
               <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Active Exc.</span>
                <AlertCircle className={`w-3 h-3 ${kpis.exceptions > 0 ? 'text-destructive' : 'text-success'}`} />
              </div>
              <p className={`text-2xl font-heading font-bold ${kpis.exceptions > 0 ? 'text-destructive' : 'text-success'}`}>
                {kpis.exceptions}
              </p>
            </div>

            <div className="glass-card p-4 rounded-xl">
               <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Step</span>
                <Terminal className="w-3 h-3 text-muted-foreground" />
              </div>
              <p className="text-2xl font-heading font-bold text-foreground">
                {kpis.steps} <span className="text-xs text-muted-foreground">/ {maxSteps}</span>
              </p>
            </div>
          </div>

          {/* Hacker Terminal */}
          <div className="glass-card flex-1 min-h-[300px] rounded-2xl flex flex-col overflow-hidden relative group">
            <div className="bg-black/40 border-b border-white/5 py-2 px-4 flex items-center gap-2">
               <Terminal className="w-4 h-4 text-primary" />
               <span className="text-[10px] font-mono tracking-widest text-primary/80 uppercase">Overseer Action Log</span>
            </div>
            
            <div className="flex-1 bg-[#0a0a0c] p-4 font-mono text-[11px] sm:text-xs overflow-y-auto leading-relaxed custom-scrollbar">
              {actionLog.length === 0 ? (
                <div className="text-white/20 italic">Awaiting uplink...</div>
              ) : (
                actionLog.map((log, idx) => {
                  let color = "text-white/70";
                  if (log.includes("V:+") && !log.includes("V:+0.00")) color = "text-success";
                  if (log.includes("V:-")) color = "text-destructive";
                  if (log.includes("[SYS]")) color = "text-primary font-bold";
                  
                  return (
                    <div key={idx} className={`${color} mb-1.5 break-words`}>
                      <span className="opacity-50 mr-2">{'>'}</span> {log}
                    </div>
                  )
                })
              )}
              <div ref={logEndRef} />
            </div>

            {/* Scanline overlay */}
            <div className="absolute inset-0 pointer-events-none opacity-[0.03] group-hover:opacity-[0.05] transition-opacity duration-500" 
                 style={{ backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, #fff 2px, #fff 4px)' }}>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default SimulationDashboard;
