function Update-IrodoriModelSelection {
    param($Dialog)

    $Selected = @($Dialog.Tag.Options | Where-Object { $_.CheckBox.Checked })
    $TotalBytes = 0.0
    foreach ($Option in $Selected) { $TotalBytes += $Option.Bytes }
    $Dialog.Tag.Summary.Text = '選択中: {0} モデル / モデル容量合計: 約 {1:N2} GB' -f $Selected.Count, ($TotalBytes / 1e9)
    $Dialog.Tag.StartButton.Enabled = $Selected.Count -gt 0
}

function New-IrodoriModelSelectionForm {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)

    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [System.Windows.Forms.Application]::EnableVisualStyles()

    $Catalog = @(
        @{ Name = 'V4.1 Small'; Repo = 'Aratako/Irodori-TTS-v4.1-Small'; Bytes = 3064295596; Description = '標準モデル。参照音声とVoiceDesignの両方に対応。' },
        @{ Name = 'V4 Large'; Repo = 'Aratako/Irodori-TTS-v4-Large'; Bytes = 13152893348; Description = '大型モデル。実行時はSmallより多くのメモリが必要です。' },
        @{ Name = 'V4.1 Anime'; Repo = 'phasefield-audio/Irodori-TTS-v4.1-Anime'; Bytes = 3064295596; Description = 'V4.1 Smallをアニメ調の音声で追加学習した派生モデル。' }
    )
    $Installed = @()
    $RecordPath = Join-Path $ProjectRoot 'model_data\installed_models.json'
    if (Test-Path -LiteralPath $RecordPath) {
        try {
            $Records = Get-Content -LiteralPath $RecordPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $Installed = @($Records | Where-Object {
                $_.path -and (Test-Path -LiteralPath (Join-Path $_.path 'model.safetensors'))
            } | ForEach-Object { $_.repo_id })
        } catch {
            # An absent or unreadable registry does not prevent a fresh setup.
            $Installed = @()
        }
    }

    $Dialog = New-Object System.Windows.Forms.Form
    $Dialog.Name = 'IrodoriSetupModelSelection'
    $Dialog.Text = 'Irodori-TTS V4 セットアップ'
    $Dialog.ClientSize = New-Object System.Drawing.Size(680, 440)
    $Dialog.StartPosition = 'CenterScreen'
    $Dialog.FormBorderStyle = 'FixedDialog'
    $Dialog.MaximizeBox = $false
    $Dialog.MinimizeBox = $false
    $Dialog.Font = New-Object System.Drawing.Font('Yu Gothic UI', 10)

    $Title = New-Object System.Windows.Forms.Label
    $Title.Text = 'ダウンロードするモデルを選択'
    $Title.Font = New-Object System.Drawing.Font('Yu Gothic UI', 14, [System.Drawing.FontStyle]::Bold)
    $Title.SetBounds(20, 16, 640, 34)
    $Dialog.Controls.Add($Title)

    $Description = New-Object System.Windows.Forms.Label
    $Description.Text = "複数選択できます。導入済みのファイルは再利用します。`nセットアップ後はWebUIのチェックポイント欄でモデルを切り替えられます。"
    $Description.SetBounds(20, 58, 640, 48)
    $Dialog.Controls.Add($Description)

    $Group = New-Object System.Windows.Forms.GroupBox
    $Group.Text = 'モデル（通常精度版）'
    $Group.SetBounds(20, 112, 640, 190)
    $Dialog.Controls.Add($Group)

    $Options = @()
    for ($Index = 0; $Index -lt $Catalog.Count; $Index++) {
        $Model = $Catalog[$Index]
        $CheckBox = New-Object System.Windows.Forms.CheckBox
        $CheckBox.Name = 'ModelOption' + $Index
        $CheckBox.Text = '{0}  /  約 {1:N2} GB' -f $Model.Name, ($Model.Bytes / 1e9)
        if ($Installed -contains $Model.Repo) { $CheckBox.Text += '  ［導入済み］' }
        $CheckBox.SetBounds(16, (27 + $Index * 52), 604, 26)
        $CheckBox.Checked = $Index -eq 0
        $CheckBox.Add_CheckedChanged({
            param($Sender, $EventArgs)
            Update-IrodoriModelSelection -Dialog $Sender.FindForm()
        })
        $Group.Controls.Add($CheckBox)

        $Detail = New-Object System.Windows.Forms.Label
        $Detail.Text = $Model.Description
        $Detail.ForeColor = [System.Drawing.Color]::DimGray
        $Detail.Font = New-Object System.Drawing.Font('Yu Gothic UI', 9)
        $Detail.SetBounds(36, (52 + $Index * 52), 584, 22)
        $Group.Controls.Add($Detail)
        $Options += [PSCustomObject]@{ Repo = $Model.Repo; Bytes = $Model.Bytes; CheckBox = $CheckBox }
    }

    $Location = New-Object System.Windows.Forms.Label
    $Location.Text = "保存先: $(Join-Path $ProjectRoot 'model_data\huggingface\hub')"
    $Location.SetBounds(20, 314, 640, 38)
    $Dialog.Controls.Add($Location)

    $Summary = New-Object System.Windows.Forms.Label
    $Summary.SetBounds(20, 354, 640, 26)
    $Dialog.Controls.Add($Summary)

    $StartButton = New-Object System.Windows.Forms.Button
    $StartButton.Name = 'StartSetup'
    $StartButton.Text = 'セットアップ開始'
    $StartButton.SetBounds(384, 392, 140, 32)
    $Dialog.Controls.Add($StartButton)
    $StartButton.Add_Click({
        param($Sender, $EventArgs)
        $Form = $Sender.FindForm()
        $Form.Tag.SelectedRepos = @($Form.Tag.Options | Where-Object { $_.CheckBox.Checked } | ForEach-Object { $_.Repo })
        $Form.DialogResult = [System.Windows.Forms.DialogResult]::OK
    })

    $CancelButton = New-Object System.Windows.Forms.Button
    $CancelButton.Text = 'キャンセル'
    $CancelButton.SetBounds(536, 392, 124, 32)
    $CancelButton.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $Dialog.Controls.Add($CancelButton)
    $Dialog.AcceptButton = $StartButton
    $Dialog.CancelButton = $CancelButton
    $Dialog.Tag = [PSCustomObject]@{
        Options = $Options; Summary = $Summary; StartButton = $StartButton; SelectedRepos = @()
    }
    Update-IrodoriModelSelection -Dialog $Dialog
    return $Dialog
}

function Select-IrodoriSetupModels {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)

    $Dialog = New-IrodoriModelSelectionForm -ProjectRoot $ProjectRoot
    try {
        if ($Dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            return $Dialog.Tag.SelectedRepos
        }
    } finally {
        $Dialog.Dispose()
    }
}
