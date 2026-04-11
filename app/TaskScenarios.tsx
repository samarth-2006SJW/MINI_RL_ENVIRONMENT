import { Zap, Target, Flame } from "lucide-react";

const scenarios = [
  {
    icon: Zap,
    level: "Easy",
    color: "text-success",
    bgColor: "bg-success/10 border-success/20",
    description: "Single shipment delay → locate component, reroute order",
    criteria: "All exceptions resolved, inventory above threshold",
  },
  {
    icon: Target,
    level: "Medium",
    color: "text-warning",
    bgColor: "bg-warning/10 border-warning/20",
    description: "Inventory shortage → verify stock, restock, notify",
    criteria: "Easy criteria + all inventory above medium threshold",
  },
  {
    icon: Flame,
    level: "Hard",
    color: "text-destructive",
    bgColor: "bg-destructive/10 border-destructive/20",
    description: "Cascading robot failure → dispatch maintenance, clear aisle, reroute",
    criteria: "Medium criteria + all robots active, battery above threshold",
  },
];

const TaskScenarios = () => {
  return (
    <section className="px-8 pb-16">
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-border/50 bg-secondary/50 text-xs text-muted-foreground mb-4">
          🏆 Evaluation
        </div>
        <h2 className="font-heading text-2xl font-bold text-foreground">
          Task Scenarios
        </h2>
        <p className="text-muted-foreground text-sm mt-2">
          Three escalating difficulty levels for agent evaluation
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {scenarios.map((s) => (
          <div
            key={s.level}
            className={`glass-card p-5 border ${s.bgColor} hover:scale-[1.02] transition-transform duration-300`}
          >
            <div className="flex items-center gap-2 mb-3">
              <s.icon className={`w-5 h-5 ${s.color}`} />
              <h3 className={`font-heading font-bold text-lg ${s.color}`}>
                {s.level}
              </h3>
            </div>
            <p className="text-sm text-foreground mb-3">{s.description}</p>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1 font-medium">
              Success Criteria
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">{s.criteria}</p>
          </div>
        ))}
      </div>
    </section>
  );
};

export default TaskScenarios;
