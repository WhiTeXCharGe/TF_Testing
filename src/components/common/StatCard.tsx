interface StatCardProps {
  label: string;
  value: React.ReactNode;
}

export function StatCard({ label, value }: StatCardProps) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}
