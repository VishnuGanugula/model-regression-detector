import React from 'react';

const categoryStyles = {
  billing: {
    bg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300',
    badgeBg: 'bg-emerald-500/20 text-emerald-200 border-emerald-400/40',
    icon: '💳',
    label: 'Billing & Payments',
    description: 'Queries regarding invoices, double charges, refunds, or payment methods.'
  },
  technical: {
    bg: 'bg-indigo-500/10 border-indigo-500/30 text-indigo-300',
    badgeBg: 'bg-indigo-500/20 text-indigo-200 border-indigo-400/40',
    icon: '⚙️',
    label: 'Technical Support',
    description: 'System bugs, login issues, broken features, or error messages.'
  },
  account: {
    bg: 'bg-amber-500/10 border-amber-500/30 text-amber-300',
    badgeBg: 'bg-amber-500/20 text-amber-200 border-amber-400/40',
    icon: '👤',
    label: 'Account Management',
    description: 'Profile settings, password resets, subscription tier changes, or security.'
  },
  general: {
    bg: 'bg-purple-500/10 border-purple-500/30 text-purple-300',
    badgeBg: 'bg-purple-500/20 text-purple-200 border-purple-400/40',
    icon: '💬',
    label: 'General Inquiry',
    description: 'General questions, feedback, feature requests, or general help.'
  },
  unknown: {
    bg: 'bg-slate-500/10 border-slate-500/30 text-slate-300',
    badgeBg: 'bg-slate-500/20 text-slate-200 border-slate-400/40',
    icon: '❓',
    label: 'Unclassified / Unknown',
    description: 'The model could not confidently categorize this ticket.'
  }
};

export default function ResultBadge({ result, onClose }) {
  if (!result) return null;

  const categoryKey = (result.predictedCategory || 'unknown').toLowerCase();
  const config = categoryStyles[categoryKey] || categoryStyles.unknown;

  return (
    <div className={`mt-6 p-5 rounded-xl border glass-card transition-all duration-300 transform animate-fade-in ${config.bg}`}>
      <div className="flex items-start justify-between">
        <div className="flex items-center space-x-3">
          <span className="text-3xl">{config.icon}</span>
          <div>
            <div className="flex items-center space-x-3">
              <h3 className="text-lg font-semibold text-white">Classification Result</h3>
              <span className={`px-3 py-1 text-xs font-bold uppercase tracking-wider rounded-full border ${config.badgeBg}`}>
                {categoryKey}
              </span>
            </div>
            <p className="text-sm mt-1 opacity-90">{config.description}</p>
          </div>
        </div>
        {onClose && (
          <button 
            onClick={onClose}
            className="text-slate-400 hover:text-white text-sm font-semibold transition-colors"
          >
            ✕
          </button>
        )}
      </div>

      <div className="mt-4 pt-3 border-t border-slate-700/50 flex justify-between items-center text-xs text-slate-400">
        <span>Recorded DB Record ID: <strong className="text-slate-200">#{result.id || 'N/A'}</strong></span>
        <span>Timestamp: {result.createdAt ? new Date(result.createdAt).toLocaleString() : 'Just now'}</span>
      </div>
    </div>
  );
}
