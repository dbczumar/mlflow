/**
 * Global MLflow Assistant Layout component.
 * Wraps main content with split-view layout when assistant panel is open.
 */

import { Global } from '@emotion/react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useCallback, useRef, useState, type ReactNode } from 'react';

import { useGlobalClaudeOptional } from './GlobalClaudeContext';
import { GlobalClaudeChatPanel } from './GlobalClaudeChatPanel';

interface GlobalClaudeLayoutProps {
  children: ReactNode;
}

const MIN_PANEL_WIDTH = 300;
const MAX_PANEL_WIDTH_PERCENT = 60;
const DEFAULT_PANEL_WIDTH_PERCENT = 34;

/**
 * Layout wrapper that provides split-view when assistant panel is open.
 * - Main content takes remaining space when panel is open, 100% otherwise
 * - Assistant panel is resizable on the right side (fixed position to appear above modals)
 */
export const GlobalClaudeLayout = ({ children }: GlobalClaudeLayoutProps) => {
  const { theme } = useDesignSystemTheme();
  const globalClaude = useGlobalClaudeOptional();
  const [panelWidth, setPanelWidth] = useState(DEFAULT_PANEL_WIDTH_PERCENT);
  const isDraggingRef = useRef(false);

  const isPanelOpen = globalClaude?.isPanelOpen ?? false;

  // Show panel when open - setup wizard will be displayed if not configured
  const showPanel = isPanelOpen;
  const contentWidth = 100 - panelWidth;

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDraggingRef.current = true;

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;

      const windowWidth = window.innerWidth;
      const newWidth = ((windowWidth - e.clientX) / windowWidth) * 100;

      // Clamp width between min and max
      const minPercent = (MIN_PANEL_WIDTH / windowWidth) * 100;
      const clampedWidth = Math.max(minPercent, Math.min(MAX_PANEL_WIDTH_PERCENT, newWidth));
      setPanelWidth(clampedWidth);
    };

    const handleMouseUp = () => {
      isDraggingRef.current = false;
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, []);

  return (
    <>
      {/* Global styles to constrain modals when Claude panel is open */}
      {showPanel && (
        <Global
          styles={{
            // Constrain Ant Design modals
            '.ant-modal-wrap': {
              right: `${panelWidth}% !important`,
              width: `${contentWidth}% !important`,
            },
            '.ant-modal-centered': {
              right: `${panelWidth}% !important`,
              width: `${contentWidth}% !important`,
            },
            // Constrain Dubois/Design System modals
            '[class*="du-bois-light-modal-wrap"], [class*="du-bois-dark-modal-wrap"]': {
              right: `${panelWidth}% !important`,
              width: `${contentWidth}% !important`,
            },
            // Generic modal backdrop adjustment
            '.ReactModal__Overlay': {
              right: `${panelWidth}% !important`,
              width: `${contentWidth}% !important`,
            },
          }}
        />
      )}
      <div
        css={{
          display: 'flex',
          height: '100%',
          overflow: 'hidden',
        }}
      >
        {/* Main content area - shrinks when panel is open */}
        <div
          css={{
            flex: showPanel ? `0 0 ${contentWidth}%` : 1,
            minWidth: 0,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            transition: isDraggingRef.current ? 'none' : 'flex 0.2s ease-in-out',
          }}
        >
          {children}
        </div>

        {/* Claude panel on the right - fixed position to appear above modals */}
        {showPanel && (
          <div
            css={{
              position: 'fixed',
              top: 0,
              right: 0,
              width: `${panelWidth}%`,
              height: '100vh',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              backgroundColor: theme.colors.backgroundPrimary,
              zIndex: 2200,
              boxShadow: '-4px 0 12px rgba(0, 0, 0, 0.15)',
            }}
          >
            {/* Resize handle */}
            <div
              onMouseDown={handleMouseDown}
              css={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: 4,
                cursor: 'col-resize',
                backgroundColor: 'transparent',
                borderLeft: `1px solid ${theme.colors.border}`,
                '&:hover': {
                  backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                },
                zIndex: 1,
              }}
            />
            <GlobalClaudeChatPanel />
          </div>
        )}
      </div>
    </>
  );
};
