import { createTheme } from '@mui/material/styles';

export const appTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2563eb',
      light: '#60a5fa',
      dark: '#1d4ed8',
    },
    secondary: {
      main: '#7c3aed',
      light: '#a78bfa',
      dark: '#6d28d9',
    },
    background: {
      default: '#f5f7fb',
      paper: '#ffffff',
    },
    text: {
      primary: '#0f172a',
      secondary: '#64748b',
    },
    success: {
      main: '#059669',
    },
    warning: {
      main: '#d97706',
    },
    error: {
      main: '#dc2626',
    },
  },
  shape: {
    borderRadius: 8,
  },
  typography: {
    fontFamily: `Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`,
    h4: {
      fontWeight: 900,
      letterSpacing: 0,
    },
    h6: {
      fontWeight: 800,
    },
    button: {
      textTransform: 'none',
      fontWeight: 800,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          border: '1px solid #e5e7eb',
          boxShadow: '0 14px 32px rgba(15, 23, 42, 0.07)',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 800,
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        head: {
          backgroundColor: '#f8fafc',
          color: '#64748b',
          fontSize: 11,
          fontWeight: 900,
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
        },
      },
    },
  },
});
