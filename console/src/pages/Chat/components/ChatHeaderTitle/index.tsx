import React, { useEffect, useRef, useState } from "react";
import { useChatAnywhereSessionsState } from "@agentscope-ai/chat";
import { useCodingMode } from "../../../../stores/codingModeStore";
import { Popover } from "antd";
import { useTranslation } from "react-i18next";
import styles from "./index.module.less";

const ChatHeaderTitle: React.FC = () => {
  const { sessions, currentSessionId } = useChatAnywhereSessionsState();
  const { codingMode } = useCodingMode();
  const { t } = useTranslation();
  const currentSession = sessions.find((s) => s.id === currentSessionId);
  const chatName = currentSession?.name || t("chat.newChat", "New Chat");

  const containerRef = useRef<HTMLSpanElement | null>(null);
  const measureRef = useRef<HTMLSpanElement | null>(null);
  const [shouldMarquee, setShouldMarquee] = useState(false);
  const [popoverOpen, setPopoverOpen] = useState(false);

  useEffect(() => {
    const check = () => {
      const containerWidth =
        containerRef.current?.getBoundingClientRect().width ?? 0;
      const textWidth =
        measureRef.current?.getBoundingClientRect().width ?? 0;
      // Marquee when text overflows its container (any screen size)
      setShouldMarquee(textWidth > containerWidth + 2);
    };

    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, [chatName, codingMode]);

  const handleSessionSwitch = (sessionId: string) => {
    if (sessionId === currentSessionId) {
      setPopoverOpen(false);
      return;
    }
    window.dispatchEvent(
      new CustomEvent("qwenpaw:sidebar-select-session", {
        detail: { sessionId },
      }),
    );
    setPopoverOpen(false);
  };

  const sortedSessions = [...sessions].sort((a, b) => {
    if (a.pinned && !b.pinned) return -1;
    if (!a.pinned && b.pinned) return 1;
    const aTime = a.updatedAt ?? a.createdAt ?? 0;
    const bTime = b.updatedAt ?? b.createdAt ?? 0;
    return new Date(bTime).getTime() - new Date(aTime).getTime();
  });

  const popoverContent = (
    <div className={styles.sessionPopover}>
      <div className={styles.sessionPopoverList}>
        {sortedSessions.map((session) => (
          <div
            key={session.id}
            className={`${styles.sessionPopoverItem} ${
              session.id === currentSessionId
                ? styles.sessionPopoverItemActive
                : ""
            }`}
            onClick={() => handleSessionSwitch(session.id)}
          >
            <span className={styles.sessionPopoverItemName}>
              {session.name || t("chat.newChat", "New Chat")}
            </span>
            {session.pinned && (
              <span className={styles.sessionPopoverItemPin}>📌</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  const className = codingMode
    ? `${styles.chatName} ${styles.chatNameCoding}`
    : styles.chatName;

  return (
    <Popover
      content={popoverContent}
      trigger="click"
      open={popoverOpen}
      onOpenChange={setPopoverOpen}
      placement="bottomLeft"
      overlayClassName={styles.sessionPopoverOverlay}
      getPopupContainer={() => document.body}
    >
      <span
        className={className}
        ref={containerRef}
        style={{ cursor: "pointer" }}
      >
        <span
          ref={measureRef}
          aria-hidden="true"
          style={{
            position: "absolute",
            visibility: "hidden",
            whiteSpace: "nowrap",
            pointerEvents: "none",
          }}
        >
          {chatName}
        </span>
        {shouldMarquee ? (
          <span className={styles.marquee}>{chatName}</span>
        ) : (
          chatName
        )}
      </span>
    </Popover>
  );
};

export default ChatHeaderTitle;
