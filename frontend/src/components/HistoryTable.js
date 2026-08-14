import React, { useState } from 'react';

const categoryBadgeStyles = {
  billing: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  technical: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
  account: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
  general: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
  unknown: 'bg-slate-500/20 text-slate-300 border-slate-500/30'
};

export default function HistoryTable({ history, loading, onRefresh }) {
  const [search, setSearch] = useState('');
  const [selectedItem, setSelectedItem] = useState(null);

  const filteredHistory = history.filter(item => {
    const textMatch = (item.emailText || '').toLowerCase().includes(search.toLowerCase());
    const catMatch = (item.predictedCategory || '').toLowerCase().includes(search.toLowerCase());
    return textMatch || catMatch;
  });

  return (
    <div className="glass-card rounded-2xl p-6 shadow-xl border border-slate-800 mt-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span>📜</span> Classification History
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Real-time record of all predictions stored in MySQL database ({history.length} records).
          </p>
        </div>

        <div className="flex items-center gap-3">
          <input
            type="text"
            placeholder="Filter history..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:border-indigo-500 w-48"
          />
          <button
            onClick={onRefresh}
            disabled={loading}
            className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded-lg text-xs font-medium border border-slate-700 flex items-center gap-1 transition-all"
          >
            <span>🔄</span> Refresh
          </button>
        </div>
      </div>

      {loading && history.length === 0 ? (
        <div className="text-center py-12 text-slate-500 text-sm">
          Loading classification history...
        </div>
      ) : filteredHistory.length === 0 ? (
        <div className="text-center py-12 border border-dashed border-slate-800 rounded-xl">
          <p className="text-slate-400 text-sm">No classification records found.</p>
          <p className="text-xs text-slate-500 mt-1">Classify your first email above to record prediction history.</p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-900/90 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="py-3 px-4 w-16">ID</th>
                <th className="py-3 px-4">Email Text Snippet</th>
                <th className="py-3 px-4 w-32">Predicted Category</th>
                <th className="py-3 px-4 w-44">Created At</th>
                <th className="py-3 px-4 w-20 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
              {filteredHistory.map((row) => {
                const categoryKey = (row.predictedCategory || 'unknown').toLowerCase();
                const badgeStyle = categoryBadgeStyles[categoryKey] || categoryBadgeStyles.unknown;

                return (
                  <tr key={row.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3.5 px-4 font-mono font-bold text-slate-400">
                      #{row.id}
                    </td>
                    <td className="py-3.5 px-4 max-w-md truncate text-slate-200">
                      {row.emailText}
                    </td>
                    <td className="py-3.5 px-4">
                      <span className={`px-2.5 py-1 text-[10px] font-bold uppercase rounded-md border ${badgeStyle}`}>
                        {categoryKey}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-slate-400 font-mono text-[11px]">
                      {row.createdAt ? new Date(row.createdAt).toLocaleString() : 'N/A'}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <button
                        onClick={() => setSelectedItem(row)}
                        className="text-indigo-400 hover:text-indigo-300 underline font-medium"
                      >
                        View
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Modal for full details */}
      {selectedItem && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-fade-in">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4">
            <div className="flex justify-between items-center border-b border-slate-800 pb-3">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <span>📋</span> Record Details #{selectedItem.id}
              </h3>
              <button 
                onClick={() => setSelectedItem(null)}
                className="text-slate-400 hover:text-white font-bold text-lg"
              >
                ✕
              </button>
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-400 block mb-1">Predicted Category:</label>
              <span className={`inline-block px-3 py-1 text-xs font-bold uppercase rounded-md border ${categoryBadgeStyles[selectedItem.predictedCategory?.toLowerCase()] || categoryBadgeStyles.unknown}`}>
                {selectedItem.predictedCategory}
              </span>
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-400 block mb-1">Full Email Content:</label>
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-xs text-slate-200 leading-relaxed max-h-48 overflow-y-auto whitespace-pre-wrap">
                {selectedItem.emailText}
              </div>
            </div>

            <div className="text-xs text-slate-500 pt-2 flex justify-between border-t border-slate-800">
              <span>DB Primary Key ID: {selectedItem.id}</span>
              <span>Saved At: {selectedItem.createdAt}</span>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setSelectedItem(null)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-xs font-semibold"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
