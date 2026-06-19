import React, { useEffect, useRef, useState } from "react";
import { IconButton } from "@agentscope-ai/design";
import {
  SparkHistoryLine,
  SparkNewChatFill,
  SparkSearchLine,
} from "@agentscope-ai/icons";
import {
  ExpandAltOutlined,
  CompressOutlined,
  MoreOutlined,
} from "@ant-design/icons";
import { useChatAnywhereSessions } from "@agentscope-ai/chat";
import { useTranslation } from "react-i18next";
import { Dropdown, Flex, Tooltip } from "antd";
import ChatSessionDrawer from "../ChatSessionDrawer";
import ChatSearchPanel from "../ChatSearchPanel";
import PlanPanel from "../../../../components/PlanPanel";
import styles from "./index.module.less";

const PlanIcon = () => (
  <svg
    width="1em"
    height="1em"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M9 11l3 3L22 4" />
    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
  </svg>
);

const PINNED_STORAGE_KEY = "qwenpaw_history_drawer_pinned";

// Below this *available header width*, collapse non-essential actions into
// a "more" dropdown. Empirically tuned for Chat header on mobile screens.
const COMPACT_BREAKPOINT_PX = 480;

interface ChatActionGroupProps {
  planEnabled?: boolean;
  isWideMode?: boolean;
  onToggleWideMode?: () => void;
  /**
   * Optional external controller for the chat history drawer.
   * When provided, callers can drive the drawer from outside (e.g. from a
   * pinned full-mode panel host). When not provided, the component falls
   * back to its internal state.
   */
  onToggleHistory?: () => void;
  historyOpen?: boolean;
}

const ChatActionGroup: React.FC<ChatActionGroupProps> = ({
  planEnabled = false,
  isWideMode = false,
  onToggleWideMode,
  onToggleHistory,
  historyOpen: historyOpenProp,
}) => {
  const { t } = useTranslation();

  const [historyPinned, setHistoryPinned] = useState(() => {
    try {
      return localStorage.getItem(PINNED_STORAGE_KEY) === "true";
    } catch {
      return false;
    }
  });

  // If pinned, auto-open drawer on mount. Otherwise honour the external
  // `historyOpen` prop when the parent provides one.
  const [historyOpenInternal, setHistoryOpenInternal] = useState(
    historyOpenProp ?? historyPinned,
  );
  const isHistoryControlled = historyOpenProp !== undefined;
  const historyOpen = isHistoryControlled
    ? historyOpenProp
    : historyOpenInternal;
  const setHistoryOpen = (open: boolean) => {
    if (isHistoryControlled) {
      // Parent drives the open state; if it also wants notifications when
      // we'd toggle, surface it via onToggleHistory below.
      if (open !== historyOpenProp) onToggleHistory?.();
    } else {
      setHistoryOpenInternal(open);
    }
  };

  const handlePinChange = (pinned: boolean) => {
    setHistoryPinned(pinned);
    try {
      if (pinned) {
        localStorage.setItem(PINNED_STORAGE_KEY, "true");
      } else {
        localStorage.removeItem(PINNED_STORAGE_KEY);
      }
    } catch {
      // storage full or unavailable
    }
  };

  const [searchOpen, setSearchOpen] = useState(false);
  const [planOpen, setPlanOpen] = useState(false);
  const { createSession } = useChatAnywhereSessions();

  // Detect compact mode by observing the parent header width.
  // Falls back to window inner width on environments without ResizeObserver.
  const groupRef = useRef<HTMLDivElement | null>(null);
  const [isCompact, setIsCompact] = useState(false);

  useEffect(() => {
    const computeCompact = () => {
      const headerEl = groupRef.current?.parentElement ?? null;
      const headerWidth =
        headerEl?.getBoundingClientRect().width ?? window.innerWidth;
      setIsCompact(headerWidth < COMPACT_BREAKPOINT_PX);
    };

    computeCompact();

    let observer: ResizeObserver | null = null;
    const headerEl = groupRef.current?.parentElement ?? null;
    if (typeof ResizeObserver !== "undefined" && headerEl) {
      observer = new ResizeObserver(() => computeCompact());
      observer.observe(headerEl);
    }

    window.addEventListener("resize", computeCompact);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", computeCompact);
    };
  }, []);

  const wideModeTooltip = isWideMode
    ? t("chat.normalModeTooltip")
    : t("chat.wideModeTooltip");

  // Items moved into the "more" dropdown when compact.
  const moreMenuItems = [
    planEnabled
      ? {
          key: "plan",
          label: t("plan.title", "Plan"),
          icon: <PlanIcon />,
          onClick: () => setPlanOpen(true),
        }
      : null,
    {
      key: "history",
      label: t("chat.chatHistoryTooltip"),
      icon: <SparkHistoryLine size={16} />,
      onClick: () => setHistoryOpen(true),
    },
    onToggleWideMode
      ? {
          key: "wideMode",
          label: wideModeTooltip,
          icon: isWideMode ? <CompressOutlined /> : <ExpandAltOutlined />,
          onClick: onToggleWideMode,
        }
      : null,
  ].filter(Boolean) as {
    key: string;
    label: string;
    icon: React.ReactNode;
    onClick: () => void;
  }[];

  return (
    <Flex
      ref={groupRef}
      gap={isCompact ? 4 : 8}
      align="center"
      className={styles.chatActionGroup}
    >
      {/* Plan button: shown inline only when not compact */}
      {planEnabled && !isCompact && (
        <Tooltip title={t("plan.title", "Plan")} mouseEnterDelay={0.5}>
          <IconButton
            bordered={false}
            icon={<PlanIcon />}
            onClick={() => setPlanOpen(true)}
          />
        </Tooltip>
      )}
      {/* Always visible: New Chat */}
      <Tooltip title={t("chat.newChatTooltip")} mouseEnterDelay={0.5}>
        <IconButton
          bordered={false}
          icon={<SparkNewChatFill />}
          onClick={() => createSession()}
        />
      </Tooltip>
      {/* Always visible: Search */}
      <Tooltip title={t("chat.searchTooltip")} mouseEnterDelay={0.5}>
        <IconButton
          bordered={false}
          icon={<SparkSearchLine />}
          onClick={() => setSearchOpen(true)}
        />
      </Tooltip>
      {/* History: shown inline only when not compact */}
      {!isCompact && (
        <Tooltip title={t("chat.chatHistoryTooltip")} mouseEnterDelay={0.5}>
          <IconButton
            bordered={false}
            icon={<SparkHistoryLine />}
            onClick={() => setHistoryOpen(true)}
          />
        </Tooltip>
      )}
      {/* Wide mode: shown inline only when not compact */}
      {onToggleWideMode && !isCompact && (
        <Tooltip title={wideModeTooltip} mouseEnterDelay={0.5}>
          <IconButton
            bordered={false}
            icon={isWideMode ? <CompressOutlined /> : <ExpandAltOutlined />}
            onClick={onToggleWideMode}
          />
        </Tooltip>
      )}
      {/* Compact mode: collect remaining actions in a "more" dropdown */}
      {isCompact && moreMenuItems.length > 0 && (
        <Dropdown
          menu={{ items: moreMenuItems }}
          placement="bottomRight"
          trigger={["click"]}
        >
          <span className={styles.moreAction}>
            <Tooltip title={t("common.more", "More")} mouseEnterDelay={0.5}>
              <IconButton bordered={false} icon={<MoreOutlined />} />
            </Tooltip>
          </span>
        </Dropdown>
      )}
      <ChatSessionDrawer
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
        pinned={historyPinned}
        onPinChange={handlePinChange}
      />
      <ChatSearchPanel open={searchOpen} onClose={() => setSearchOpen(false)} />
      {planEnabled && (
        <PlanPanel open={planOpen} onClose={() => setPlanOpen(false)} />
      )}
    </Flex>
  );
};

export default ChatActionGroup;
