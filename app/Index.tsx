import { useState } from "react";
import DashboardSidebar from "./DashboardSidebar";
import HeroSection from "./HeroSection";
import FeaturesGrid from "./FeaturesGrid";
import TaskScenarios from "./TaskScenarios";
import SimulationDashboard from "./SimulationDashboard";
import ApiDocs from "./ApiDocs";

const Index = () => {
  const [activeSection, setActiveSection] = useState("home");

  const renderSection = () => {
    switch (activeSection) {
      case "simulate":
        return <SimulationDashboard />;
      case "docs":
        return <ApiDocs />;
      default:
        return (
          <>
            <HeroSection onRunEpisode={() => setActiveSection("simulate")} />
            <FeaturesGrid />
            <TaskScenarios />
          </>
        );
    }
  };

  return (
    <div className="flex min-h-screen">
      <DashboardSidebar activeSection={activeSection} onNavigate={setActiveSection} />
      <main className="ml-[220px] flex-1 overflow-y-auto">
        {renderSection()}
      </main>
    </div>
  );
};

export default Index;
