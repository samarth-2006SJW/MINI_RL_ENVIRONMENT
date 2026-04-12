import React from 'react';

interface WarehouseGridProps {
  gridState: {
    grid_size: [number, number];
    robots: any[];
    stations: any[];
    obstacles: number[][];
  } | null;
}

const WarehouseGrid = ({ gridState }: WarehouseGridProps) => {
  if (!gridState) {
    return (
      <div className="w-full aspect-video flex flex-col items-center justify-center bg-black/20 rounded-xl border border-white/5">
        <div className="w-8 h-8 rounded-full border-2 border-primary border-t-transparent animate-spin mb-4" />
        <p className="text-muted-foreground text-sm font-medium">Awaiting Grid Telemetry...</p>
      </div>
    );
  }

  const gridSize = gridState.grid_size;
  if (!Array.isArray(gridSize) || gridSize.length < 2) {
    return (
      <div className="w-full aspect-video flex items-center justify-center bg-black/20 rounded-xl border border-white/5">
        <p className="text-muted-foreground text-sm">Invalid grid dimensions</p>
      </div>
    );
  }

  const [width, height] = gridSize;

  // Pre-index robots/obstacles/stations for O(1) lookup — avoids repeated .find() crashes
  const robotMap: Record<string, any> = {};
  for (const rb of (gridState.robots || [])) {
    // Guard: skip robots with null/undefined location (SENSOR_FAILURE etc.)
    if (rb.location != null && Array.isArray(rb.location)) {
      const key = `${rb.location[0]}-${rb.location[1]}`;
      robotMap[key] = rb;
    }
  }

  const obstacleSet = new Set<string>();
  for (const o of (gridState.obstacles || [])) {
    if (Array.isArray(o) && o.length >= 2) {
      obstacleSet.add(`${o[0]}-${o[1]}`);
    }
  }

  const stationMap: Record<string, any> = {};
  for (const st of (gridState.stations || [])) {
    const loc = st?.location;
    if (loc != null && Array.isArray(loc) && loc.length >= 2) {
      stationMap[`${loc[0]}-${loc[1]}`] = st;
    }
  }

  const cells = [];
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const key = `${r}-${c}`;
      const isObstacle = obstacleSet.has(key);
      const robot = robotMap[key];
      const station = stationMap[key];

      let className = "w-5 h-5 sm:w-6 sm:h-6 md:w-8 md:h-8 flex items-center justify-center text-[10px] sm:text-xs transition-colors duration-300 ";
      let content = null;
      let title = `[${r}, ${c}]`;

      if (isObstacle) {
        className += "bg-red-900/40 border border-red-500/20";
        content = <span className="text-red-500/60 text-[8px]">╳</span>;
        title += " - Blocked";
      } else if (robot) {
        const status: string = robot.status ?? "";
        let statusColor = "bg-primary text-primary-foreground shadow-[0_0_10px_rgba(var(--primary),0.6)]";
        if (status === "MAINTENANCE") statusColor = "bg-destructive text-destructive-foreground shadow-[0_0_10px_rgba(var(--destructive),0.6)] animate-pulse";
        if (status === "SENSOR_FAILURE") statusColor = "bg-yellow-500/80 text-black shadow-[0_0_10px_rgba(234,179,8,0.5)]";

        // Safe robot label: "R1" → "R1", "ROBOT_1" → "R1"
        const robotId: string = robot.id ?? "?";
        const label = robotId.includes("_") ? `R${robotId.split("_")[1]}` : robotId;
        const battery: number = robot.battery ?? 100;

        className += `${statusColor} rounded-md z-10 font-bold relative`;
        content = (
          <>
            <span className="text-[8px] sm:text-[10px] leading-none tracking-tighter">{label}</span>
            {battery < 20 && <div className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-destructive animate-ping" />}
          </>
        );
        title += ` - ${robotId} (${battery.toFixed ? battery.toFixed(1) : battery}% batt, ${status})`;
      } else if (station) {
        className += "bg-accent/20 border border-accent/30 text-accent";
        const stationType: string = station.type || "charger";
        content = stationType === "charger" ? "⚡" : stationType === "pack" ? "📦" : "📥";
        title += ` - ${stationType.toUpperCase()}`;
      } else {
        className += "bg-transparent border border-white/5 hover:bg-white/5";
      }

      cells.push(
        <div key={key} className={className} title={title}>
          {content}
        </div>
      );
    }
  }

  return (
    <div className="p-4 bg-black/40 rounded-xl border border-white/10 flex items-center justify-center overflow-x-auto shadow-inner">
      <div
        className="grid gap-[2px] p-[2px] bg-white/5 rounded-lg"
        style={{ gridTemplateColumns: `repeat(${width}, minmax(0, 1fr))` }}
      >
        {cells}
      </div>
    </div>
  );
};

export default WarehouseGrid;
