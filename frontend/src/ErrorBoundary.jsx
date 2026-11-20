import React from 'react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '2rem', color: '#ff6b6b', fontFamily: 'sans-serif', background: '#121316', minHeight: '100vh' }}>
          <h2>Something went wrong.</h2>
          <pre style={{ background: '#1c1d22', padding: '1rem', borderRadius: '6px' }}>
            {this.state.error?.message || 'Unknown error occurred'}
          </pre>
          <button
            onClick={() => window.location.reload()}
            style={{ padding: '0.5rem 1rem', marginTop: '1rem', cursor: 'pointer' }}
          >
            Reload Page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
