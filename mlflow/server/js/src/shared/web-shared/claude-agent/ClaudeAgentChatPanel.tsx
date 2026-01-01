/**
 * Chat panel component for Claude Agent interaction.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Button,
  Input,
  Spinner,
  Typography,
  useDesignSystemTheme,
  SendIcon,
  DangerIcon,
  UserIcon,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useClaudeAgentContext } from './ClaudeAgentContext';
import type { ChatMessage } from './types';
import { GenAIMarkdownRenderer } from '../genai-markdown-renderer';

const COMPONENT_ID_PREFIX = 'mlflow.claude_agent.chat_panel';

/**
 * Single chat message bubble.
 */
const ChatMessageBubble = ({ message }: { message: ChatMessage }) => {
  const { theme } = useDesignSystemTheme();
  const isUser = message.role === 'user';

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        marginBottom: theme.spacing.md,
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          marginBottom: theme.spacing.xs,
        }}
      >
        {isUser ? (
          <UserIcon css={{ fontSize: 14, color: theme.colors.textSecondary }} />
        ) : (
          <span css={{ fontSize: 14 }}>Claude</span>
        )}
        <Typography.Text size="sm" color="secondary">
          {message.timestamp.toLocaleTimeString()}
        </Typography.Text>
      </div>
      <div
        css={{
          maxWidth: '85%',
          padding: theme.spacing.md,
          borderRadius: theme.borders.borderRadiusLg,
          backgroundColor: isUser ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.backgroundSecondary,
          color: isUser ? theme.colors.actionPrimaryTextDefault : theme.colors.textPrimary,
        }}
      >
        {isUser ? (
          <Typography.Text css={{ whiteSpace: 'pre-wrap' }}>{message.content}</Typography.Text>
        ) : (
          <GenAIMarkdownRenderer>{message.content}</GenAIMarkdownRenderer>
        )}
        {message.isStreaming && (
          <span
            css={{
              display: 'inline-block',
              width: 8,
              height: 16,
              backgroundColor: theme.colors.textPrimary,
              marginLeft: theme.spacing.xs,
              animation: 'blink 1s infinite',
              '@keyframes blink': {
                '0%, 50%': { opacity: 1 },
                '51%, 100%': { opacity: 0 },
              },
            }}
          />
        )}
      </div>
    </div>
  );
};

/**
 * Initial prompt suggestions.
 */
const PromptSuggestions = ({ onSelect }: { onSelect: (prompt: string) => void }) => {
  const { theme } = useDesignSystemTheme();

  const suggestions = [
    {
      label: 'Analyze this trace',
      prompt: 'Please analyze this trace and explain what happened, including the flow of execution.',
    },
    {
      label: 'Find issues',
      prompt: 'Identify any errors, performance issues, or potential problems in this trace.',
    },
    {
      label: 'Explain the flow',
      prompt: 'Walk me through the execution flow of this trace step by step.',
    },
    {
      label: 'Debug errors',
      prompt: 'Help me debug the errors in this trace and suggest fixes.',
    },
  ];

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        marginBottom: theme.spacing.lg,
      }}
    >
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="Get started with a suggestion:"
          description="Label above prompt suggestions in Claude chat"
        />
      </Typography.Text>
      <div
        css={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: theme.spacing.sm,
        }}
      >
        {suggestions.map((suggestion) => (
          <Button
            key={suggestion.label}
            componentId={`${COMPONENT_ID_PREFIX}.suggestion`}
            size="small"
            onClick={() => onSelect(suggestion.prompt)}
          >
            {suggestion.label}
          </Button>
        ))}
      </div>
    </div>
  );
};

/**
 * Error banner component.
 */
const ErrorBanner = ({ error, onRetry }: { error: string; onRetry?: () => void }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        padding: theme.spacing.md,
        backgroundColor: theme.colors.actionDangerPrimaryBackgroundDefault,
        borderRadius: theme.borders.borderRadiusMd,
        marginBottom: theme.spacing.md,
      }}
    >
      <DangerIcon css={{ color: theme.colors.textValidationDanger }} />
      <Typography.Text css={{ flex: 1, color: theme.colors.white }}>{error}</Typography.Text>
      {onRetry && (
        <Button componentId={`${COMPONENT_ID_PREFIX}.retry`} size="small" onClick={onRetry}>
          <FormattedMessage defaultMessage="Retry" description="Retry button text" />
        </Button>
      )}
    </div>
  );
};

/**
 * Setup required banner.
 */
const SetupRequiredBanner = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        padding: theme.spacing.lg,
        backgroundColor: theme.colors.backgroundSecondary,
        borderRadius: theme.borders.borderRadiusLg,
        marginBottom: theme.spacing.md,
      }}
    >
      <Typography.Title level={4}>
        <FormattedMessage defaultMessage="Setup Required" description="Title for setup required banner" />
      </Typography.Title>
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="Claude Code needs to be configured to use this feature. Please run the following command in your terminal:"
          description="Setup instructions for Claude agent"
        />
      </Typography.Text>
      <code
        css={{
          padding: theme.spacing.md,
          backgroundColor: theme.colors.backgroundPrimary,
          borderRadius: theme.borders.borderRadiusMd,
          fontFamily: 'monospace',
        }}
      >
        mlflow claude init
      </code>
      <Typography.Text size="sm" color="secondary">
        <FormattedMessage
          defaultMessage="This will configure Claude CLI and install the necessary skills for trace analysis."
          description="Additional setup info"
        />
      </Typography.Text>
    </div>
  );
};

/**
 * Main chat panel component.
 */
export const ClaudeAgentChatPanel = () => {
  const { theme } = useDesignSystemTheme();
  const { messages, isStreaming, error, isClaudeAvailable, startAnalysis, sendMessage } = useClaudeAgentContext();
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = useCallback(() => {
    if (inputValue.trim() && !isStreaming) {
      if (messages.length === 0) {
        startAnalysis(inputValue.trim());
      } else {
        sendMessage(inputValue.trim());
      }
      setInputValue('');
    }
  }, [inputValue, isStreaming, messages.length, startAnalysis, sendMessage]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const handleSuggestionSelect = useCallback(
    (prompt: string) => {
      startAnalysis(prompt);
    },
    [startAnalysis],
  );

  // Show setup banner if Claude is not available
  if (isClaudeAvailable === false) {
    return (
      <div css={{ padding: theme.spacing.lg }}>
        <SetupRequiredBanner />
      </div>
    );
  }

  // Show loading while checking availability
  if (isClaudeAvailable === null) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          padding: theme.spacing.lg,
        }}
      >
        <Spinner />
      </div>
    );
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
      }}
    >
      {/* Messages area */}
      <div
        css={{
          flex: 1,
          overflowY: 'auto',
          padding: theme.spacing.lg,
        }}
        onWheel={(e) => e.stopPropagation()}
      >
        {error && <ErrorBanner error={error} />}

        {messages.length === 0 && !isStreaming && <PromptSuggestions onSelect={handleSuggestionSelect} />}

        {messages.map((message) => (
          <ChatMessageBubble key={message.id} message={message} />
        ))}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div
        css={{
          borderTop: `1px solid ${theme.colors.border}`,
          padding: theme.spacing.md,
          display: 'flex',
          gap: theme.spacing.sm,
        }}
      >
        <Input
          componentId={`${COMPONENT_ID_PREFIX}.input`}
          placeholder="Ask a question about this trace..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isStreaming}
          css={{ flex: 1 }}
        />
        <Button
          componentId={`${COMPONENT_ID_PREFIX}.send`}
          type="primary"
          onClick={handleSend}
          disabled={!inputValue.trim() || isStreaming}
          icon={isStreaming ? <Spinner size="small" /> : <SendIcon />}
        >
          <FormattedMessage defaultMessage="Send" description="Send button text" />
        </Button>
      </div>
    </div>
  );
};
