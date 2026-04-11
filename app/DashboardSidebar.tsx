import { Home, Play, BookOpen, Warehouse, Settings } from "lucide-react";

// Only 3 real sections that actually work
const navItems = [
  { icon: Home,     label: "Overview",      id: "home",     desc: "Intro & scenarios" },
  { icon: Play,     label: "Simulation",    id: "simulate", desc: "Run & watch agents" },
  { icon: BookOpen, label: "API Docs",      id: "docs",     desc: "OpenEnv endpoints" },
];

interface DashboardSidebarProps {
  activeSection: string;
  onNavigate: (id: string) => void;
}

const DashboardSidebar = ({ activeSection, onNavigate }: DashboardSidebarProps) => {
  return (
    <aside className="glass-sidebar w-[220px] min-h-screen flex flex-col fixed left-0 top-0 z-30">
      {/* Logo */}
      <div className="p-5 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-lg bg-primary/20 flex items-center justify-center">
            <Warehouse className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h1 className="font-heading font-bold text-base text-foreground leading-tight">
              Warehouse <span className="text-gradient">RL</span>
            </h1>
            <p className="text-[10px] text-muted-foreground uppercase tracking-widest">
              OpenEnv · v1.0
            </p>
          </div>
        </div>
      </div>

      {/* Backend Status */}
      <div className="px-5 pb-3">
        <div className="flex items-center gap-1.5 text-xs">
          <span className="w-2 h-2 rounded-full bg-success pulse-dot" />
          <span className="text-muted-foreground">Backend:</span>
          <span className="text-success font-medium">Online</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 space-y-0.5 mt-2">
        {navItems.map((item) => {
          const isActive = activeSection === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex flex-col items-start px-3 py-2.5 rounded-lg text-sm transition-all duration-200 ${
                isActive
                  ? "bg-secondary text-foreground font-medium"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
              }`}
            >
              <div className="flex items-center gap-2.5">
                <item.icon className={`w-4 h-4 ${isActive ? "text-primary" : ""}`} />
                <span>{item.label}</span>
              </div>
              <span className={`text-[10px] pl-[26px] mt-0.5 ${isActive ? "text-muted-foreground" : "text-muted-foreground/50"}`}>
                {item.desc}
              </span>
            </button>
          );
        })}
      </nav>

      {/* Episode Reward */}
      <div className="mx-4 mb-4 glass-card p-3 glow-primary">
        <div className="flex items-center justify-between mb-1">
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
            Max Reward
          </p>
          <Settings className="w-3 h-3 text-muted-foreground" />
        </div>
        <p className="text-3xl font-heading font-bold text-primary">1.000</p>
        <p className="text-[10px] text-muted-foreground mt-0.5">per episode</p>
      </div>
    </aside>
  );
};

export default DashboardSidebar;
