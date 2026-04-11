import { Bot, AlertTriangle, Package, Route, Wrench, Brain } from "lucide-react";

const features = [
  {
    icon: Bot,
    title: "Robot Fleet\nManagement",
    description: "Monitor robot status, battery levels, and locations across the warehouse grid.",
  },
  {
    icon: AlertTriangle,
    title: "Exception\nResolution",
    description: "Handle cascading failures: breakdowns, shortages, and shipment delays in real-time.",
  },
  {
    icon: Package,
    title: "Inventory\nTracking",
    description: "Real-time component stock levels with automated restock triggers and thresholds.",
  },
  {
    icon: Route,
    title: "Path\nRerouting",
    description: "Dynamic path optimization around blocked aisles and obstructions.",
  },
  {
    icon: Wrench,
    title: "Maintenance\nDispatch",
    description: "Automated dispatch of maintenance crews for robot breakdowns and aisle clearing.",
  },
  {
    icon: Brain,
    title: "OpenEnv\nCompliance",
    description: "Standardized RL endpoints with reset/step API, supporting PPO/DQN agent integration.",
  },
];

const FeaturesGrid = () => {
  return (
    <section className="px-8 pb-12">
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-border/50 bg-secondary/50 text-xs text-muted-foreground mb-4">
          Core Capabilities
        </div>
        <h2 className="font-heading text-2xl font-bold text-foreground">
          Features You'll Actually Use
        </h2>
        <p className="text-muted-foreground text-sm mt-2">
          Practical tools designed for warehouse logistics RL scenarios
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {features.map((feature) => (
          <div
            key={feature.title}
            className="glass-card p-5 group hover:border-primary/30 transition-all duration-300"
          >
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center mb-3 group-hover:bg-primary/20 transition-colors">
              <feature.icon className="w-5 h-5 text-primary" />
            </div>
            <h3 className="font-heading font-semibold text-sm text-foreground whitespace-pre-line leading-snug mb-2">
              {feature.title}
            </h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {feature.description}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
};

export default FeaturesGrid;
