import React, { useState, useEffect, useMemo } from "react";
import { SaveOutlined } from "@ant-design/icons";
import { Select, Switch, Button } from "@agentscope-ai/design";
import type { AuxiliaryModelConfig, ProviderInfo } from "../../../../../api/types";
import api from "../../../../../api";
import { useTranslation } from "react-i18next";
import { useAppMessage } from "../../../../../hooks/useAppMessage";
import styles from "../../index.module.less";

interface AuxiliaryModelModalProps {
  open: boolean;
  providers: ProviderInfo[];
  initialConfig: AuxiliaryModelConfig | null;
  onClose: () => void;
  onSaved: () => void;
}

export const AuxiliaryModelModal = React.memo(
  function AuxiliaryModelModal({
    open,
    providers,
    initialConfig,
    onClose,
    onSaved,
  }: AuxiliaryModelModalProps) {
    const { t } = useTranslation();
    const { message } = useAppMessage();
    const [saving, setSaving] = useState(false);
    const [enabled, setEnabled] = useState(false);
    const [selectedProviderId, setSelectedProviderId] = useState<
      string | undefined
    >(undefined);
    const [selectedModel, setSelectedModel] = useState<string | undefined>(
      undefined,
    );

    useEffect(() => {
      if (open) {
        setEnabled(initialConfig?.enabled ?? false);
        setSelectedProviderId(
          initialConfig?.vision_model?.provider_id || undefined,
        );
        setSelectedModel(initialConfig?.vision_model?.model || undefined);
      }
    }, [open, initialConfig]);

    const eligible = useMemo(
      () =>
        providers.filter((p) => {
          const hasModels =
            (p.models?.length ?? 0) + (p.extra_models?.length ?? 0) > 0;
          if (!hasModels) return false;
          if (p.require_api_key === false) return !!p.base_url;
          if (p.is_custom) return !!p.base_url;
          if (p.require_api_key ?? true) return !!p.api_key;
          return true;
        }),
      [providers],
    );

    const chosenProvider = providers.find((p) => p.id === selectedProviderId);
    const modelOptions = [
      ...(chosenProvider?.models ?? []),
      ...(chosenProvider?.extra_models ?? []),
    ];
    const hasModels = modelOptions.length > 0;

    const handleProviderChange = (pid: string) => {
      setSelectedProviderId(pid);
      setSelectedModel(undefined);
    };

    const handleModelChange = (model: string) => {
      setSelectedModel(model);
    };

    const handleSave = async () => {
      if (!selectedProviderId || !selectedModel) {
        message.warning(t("models.selectModel"));
        return;
      }

      const body: AuxiliaryModelConfig = {
        enabled,
        vision_model: {
          provider_id: selectedProviderId,
          model: selectedModel,
        },
      };

      setSaving(true);
      try {
        await api.setAuxiliaryModel(body);
        message.success(t("models.visionModelSaved"));
        onSaved();
        onClose();
      } catch (error) {
        const errMsg =
          error instanceof Error
            ? error.message
            : t("models.visionModelSaveFailed");
        message.error(errMsg);
      } finally {
        setSaving(false);
      }
    };

    if (!open) return null;

    return (
      <div className={styles.auxiliaryModal}>
        <p className={styles.llmDescription}>
          {t("models.auxiliaryModelDescription")}
        </p>

        <div className={styles.slotForm}>
          <div className={styles.slotField}>
            <label className={styles.slotLabel}>
              {t("models.enableAuxiliaryModel")}
            </label>
            <Switch
              checked={enabled}
              onChange={(val) => setEnabled(val as boolean)}
            />
          </div>

          {enabled && (
            <>
              <div className={styles.slotField}>
                <label className={styles.slotLabel}>
                  {t("models.visionModelProvider")}
                </label>
                <Select
                  style={{ width: "100%" }}
                  placeholder={t("models.selectProvider")}
                  value={selectedProviderId}
                  onChange={handleProviderChange}
                  options={eligible.map((p) => ({
                    value: p.id,
                    label: p.name,
                  }))}
                />
              </div>

              <div className={styles.slotField}>
                <label className={styles.slotLabel}>
                  {t("models.visionModelModel")}
                </label>
                <Select
                  style={{ width: "100%" }}
                  placeholder={
                    hasModels
                      ? t("models.selectModel")
                      : t("models.addModelFirst")
                  }
                  disabled={!hasModels}
                  showSearch
                  optionFilterProp="label"
                  value={selectedModel}
                  onChange={handleModelChange}
                  options={modelOptions.map((m) => ({
                    value: m.id,
                    label: `${m.name} (${m.id})`,
                  }))}
                />
              </div>
            </>
          )}

          <div className={[styles.slotField, styles.slotActionField].join(" ")}>
            <Button
              type="primary"
              loading={saving}
              disabled={enabled && (!selectedProviderId || !selectedModel)}
              onClick={handleSave}
              block
              icon={<SaveOutlined />}
            >
              {t("models.save")}
            </Button>
          </div>
        </div>
      </div>
    );
  },
);
