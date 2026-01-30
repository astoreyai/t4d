{{/*
Expand the name of the chart.
*/}}
{{- define "world-weaver.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "world-weaver.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "world-weaver.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "world-weaver.labels" -}}
helm.sh/chart: {{ include "world-weaver.chart" . }}
{{ include "world-weaver.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "world-weaver.selectorLabels" -}}
app.kubernetes.io/name: {{ include "world-weaver.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
API labels
*/}}
{{- define "world-weaver.api.labels" -}}
{{ include "world-weaver.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "world-weaver.api.selectorLabels" -}}
{{ include "world-weaver.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Worker labels
*/}}
{{- define "world-weaver.worker.labels" -}}
{{ include "world-weaver.labels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Worker selector labels
*/}}
{{- define "world-weaver.worker.selectorLabels" -}}
{{ include "world-weaver.selectorLabels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "world-weaver.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "world-weaver.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Get the secret name
*/}}
{{- define "world-weaver.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- include "world-weaver.fullname" . }}-secrets
{{- end }}
{{- end }}

{{/*
Get the image tag
*/}}
{{- define "world-weaver.api.imageTag" -}}
{{- .Values.api.image.tag | default .Chart.AppVersion }}
{{- end }}

{{- define "world-weaver.worker.imageTag" -}}
{{- .Values.worker.image.tag | default .Chart.AppVersion }}
{{- end }}
