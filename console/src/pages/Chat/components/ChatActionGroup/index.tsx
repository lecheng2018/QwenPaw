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
import type { MenuProps } from "antd";
import ChatSearchPanel from "../ChatSearchPanel";
import PlanPanel from "../../../../components/PlanPanel";

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

// Below this *available header width*, collapse secondary actions (Plan,
// History, WideMode) into a "more" dropdown so the essential New/Search
// buttons stay visible on mobile.
const COMPACT_BREAKPOINT_PX = 900;

interface ChatActionGroupProps {
  planEnabled?: boolean;
  /** Callback to toggle the right-side history panel */
  onToggleHistory?: () => void;
  /** Whether the history panel is currently visible */
  historyOpen?: boolean;
  isWideMode?: boolean;
  onToggleWideMode?: () => void;
}

const ChatActionGroup: React.FC<ChatActionGroupProps> = ({
  planEnabled = false,
  onToggleHistory,
  historyOpen = false,
  isWideMode = false,
  onToggleWideMode,
}) => {
  const { t } = useTranslation();

  const [searchOpen, setSearchOpen] = useState(false);
  const [planOpen, setPlanOpen] = useState(false);
  const { createSession } = useChatAnywhereSessions();

  // Detect compact mode by viewport width.
  const groupRef = useRef<HTMLDivElement | null>(null);
  const [isCompact, setIsCompact] = useState(
    window.innerWidth < COMPACT_BREAKPOINT_PX,
  );

  useEffect(() => {
    const check = () => {
      setIsCompact(window.innerWidth < COMPACT_BREAKPOINT_PX);
    };

    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  const moreMenuItems: MenuProps["items"] = [
    {
      key: "plan",
      label: t("plan.title", "Plan"),
      icon: <PlanIcon />,
      onClick: () => setPlanOpen(true),
    },
    {
      key: "history",
      label: t("chat.chatHistoryTooltip", "Chat History"),
      icon: (
        <SparkHistoryLine
          style={
            historyOpen
              ? { color: "var(--color-primary, #ff9d4d)" }
              : undefined
          }
        />
      ),
      onClick: onToggleHistory,
    },
    {
      key: "wideMode",
      label: isWideMode
        ? t("chat.normalModeTooltip", "Normal Mode")
        : t("chat.wideModeTooltip", "Wide Mode"),
      icon: isWideMode ? <CompressOutlined /> : <ExpandAltOutlined />,
      onClick: onToggleWideMode,
    },
  ];

  // In compact mode, only show NewChat + Search as primary buttons.
  // Everything else goes into the "more" dropdown.
  return (
    <Flex gap={isCompact ? 2 : 8} align="center" ref={groupRef} style={{ flexShrink: 0 }}>
      {!isCompact && planEnabled && (
        <Tooltip title={t("plan.title", "Plan")} mouseEnterDelay={0.5}>
          <IconButton
            bordered={false}
            icon={<PlanIcon />}
            onClick={() => setPlanOpen(true)}
          />
        </Tooltip>
      )}
      <Tooltip title={t("chat.newChatTooltip")} mouseEnterDelay={0.5}>
        <IconButton
          bordered={false}
          icon={<SparkNewChatFill />}
          onClick={() => createSession()}
        />
      </Tooltip>
      <Tooltip title={t("chat.searchTooltip")} mouseEnterDelay={0.5}>
        <IconButton
          bordered={false}
          icon={<SparkSearchLine />}
          onClick={() => setSearchOpen(true)}
        />
      </Tooltip>
      {!isCompact && onToggleHistory && (
        <Tooltip title={t("chat.chatHistoryTooltip")} mouseEnterDelay={0.5}>
          <IconButton
            bordered={false}
            icon={<SparkHistoryLine />}
            style={
              historyOpen
                ? { color: "var(--color-primary, #ff9d4d)" }
                : undefined
            }
            onClick={onToggleHistory}
          />
        </Tooltip>
      )}
      {!isCompact && onToggleWideMode && (
        <Tooltip
          title={
            isWideMode ? t("chat.normalModeTooltip") : t("chat.wideModeTooltip")
          }
          mouseEnterDelay={0.5}
        >
          <IconButton
            bordered={false}
            icon={isWideMode ? <CompressOutlined /> : <ExpandAltOutlined />}
            onClick={onToggleWideMode}
          />
        </Tooltip>
      )}
      {isCompact && (
        <Dropdown menu={{ items: moreMenuItems }} trigger={["click"]}>
          <IconButton bordered={false} icon={<MoreOutlined />} />
        </Dropdown>
      )}
      <ChatSearchPanel open={searchOpen} onClose={() => setSearchOpen(false)} />
      {planEnabled && (
        <PlanPanel open={planOpen} onClose={() => setPlanOpen(false)} />
      )}
    </Flex>
  );
};

export default ChatActionGroup;
