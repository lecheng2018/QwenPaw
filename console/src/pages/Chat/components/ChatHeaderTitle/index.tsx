import React, { useEffect, useRef, useState } from "react";
import { useChatAnywhereSessionsState } from "@agentscope-ai/chat";
import { useCodingMode } from "../../../../stores/codingModeStore";
import styles from "./index.module.less";

const MOBILE_BREAKPOINT_PX = 480;

const ChatHeaderTitle: React.FC = () => {
  const { sessions, currentSessionId } = useChatAnywhereSessionsState();
  const { codingMode } = useCodingMode();
  const currentSession = sessions.find((s) => s.id === currentSessionId);
  const chatName = currentSession?.name || "New Chat";

  // Detect mobile + whether title overflows. On mobile + overflow, render as
  // a horizontal marquee; otherwise keep the original ellipsis behavior.
  const containerRef = useRef<HTMLSpanElement | null>(null);
  const measureRef = useRef<HTMLSpanElement | null>(null);
  const [shouldMarquee, setShouldMarquee] = useState(false);

  useEffect(() => {
    const check = () => {
      const w =
        typeof window !== "undefined" ? window.innerWidth : Number.MAX_VALUE;
      const isMobile = w <= MOBILE_BREAKPOINT_PX;
      if (!isMobile) {
        setShouldMarquee(false);
        return;
      }
      const containerWidth =
        containerRef.current?.getBoundingClientRect().width ?? 0;
      const textWidth =
        measureRef.current?.getBoundingClientRect().width ?? 0;
      // Add a few px tolerance to avoid borderline jitter.
      setShouldMarquee(textWidth > containerWidth + 2);
    };

    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, [chatName, codingMode]);

  const className = codingMode
    ? `${styles.chatName} ${styles.chatNameCoding}`
    : styles.chatName;

  return (
    <span className={className} ref={containerRef}>
      {/* Hidden span used to measure intrinsic text width. */}
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
  );
};

export default ChatHeaderTitle;
