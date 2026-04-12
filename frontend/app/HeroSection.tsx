import { Play, Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HeroSectionProps {
  onRunEpisode?: () => void;
}

const HeroSection = ({ onRunEpisode }: HeroSectionProps) => {
  return (
    <section className="relative py-16 px-8 text-center overflow-hidden">
      {/* Background glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[300px] bg-primary/5 rounded-full blur-[100px]" />
      </div>

      <div className="relative z-10">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-border/50 bg-secondary/50 text-xs text-muted-foreground mb-6">
          <span className="text-primary">✨</span>
          <span>OPENENV REINFORCEMENT LEARNING</span>
        </div>

        <h1 className="font-heading text-4xl md:text-5xl font-bold text-foreground leading-tight mb-4">
          Warehouse Logistics{" "}
          <span className="text-gradient">RL Environment</span>
        </h1>

        <p className="text-muted-foreground max-w-xl mx-auto text-base mb-8 leading-relaxed">
          A dynamic warehouse control system where an AI agent acts as the Central
          Dispatcher, resolving cascading logistics failures — robot breakdowns,
          inventory shortages, and shipment delays.
        </p>

        <div className="flex items-center justify-center gap-3">
          <Button 
            onClick={onRunEpisode}
            className="bg-primary text-primary-foreground hover:bg-primary/90 px-6 py-5 rounded-xl font-heading font-semibold text-sm gap-2"
          >
            <Play className="w-4 h-4" />
            Run Episode
          </Button>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
