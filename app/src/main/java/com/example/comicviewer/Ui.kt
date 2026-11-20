package com.example.comicviewer

import android.app.Activity
import android.app.Dialog
import android.graphics.Typeface
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView

/** 화면 구성에 쓰는 작은 도우미 (아래에서 올라오는 시트, 공통 크기/색) */
internal fun Activity.dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()
internal fun Activity.col(id: Int): Int = getColor(id)

internal class SheetParts(val dialog: Dialog, val root: LinearLayout, val subView: TextView, val header: LinearLayout)

/**
 * 아래에서 올라오는 시트: 손잡이 + 제목(+작은 보조 글) + 닫기. 본문은 root 에 이어서 추가.
 * heightPx 는 창 높이(WRAP_CONTENT 가능), widthPx 는 화면 너비(넓은 화면에서는 가운데 560dp 로 제한).
 * 창 배경을 시트 모양으로 두어 시스템 바 여백 영역까지 같은 색으로 채워짐.
 */
internal fun Activity.createSheet(title: String, sub: String?, widthPx: Int, heightPx: Int, headerExtra: View? = null): SheetParts {
    val dlg = Dialog(this, R.style.SheetDialog)
    val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; isFocusableInTouchMode = true }
    root.addView(View(this).apply { setBackgroundResource(R.drawable.sheet_grab) },
        LinearLayout.LayoutParams(dp(36), dp(4)).apply { gravity = Gravity.CENTER_HORIZONTAL; topMargin = dp(10); bottomMargin = dp(4) })
    val header = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL; setPadding(dp(20), dp(6), dp(10), dp(10)) }
    header.addView(TextView(this).apply { text = title; textSize = 20f; setTypeface(null, Typeface.BOLD); setTextColor(col(R.color.txt)); maxLines = 1 },
        LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
    val subView = TextView(this).apply { textSize = 12f; setTextColor(col(R.color.txt3)); setPadding(dp(10), 0, 0, 0); text = sub ?: ""; visibility = if (sub.isNullOrEmpty()) View.GONE else View.VISIBLE }
    header.addView(subView, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
    header.addView(View(this), LinearLayout.LayoutParams(0, 1, 1f)) // 남는 자리를 채워 오른쪽 버튼들을 끝으로 밈
    if (headerExtra != null) header.addView(headerExtra, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { marginEnd = dp(4) })
    val close = ImageView(this).apply {
        setImageResource(R.drawable.ic_close); imageTintList = android.content.res.ColorStateList.valueOf(col(R.color.txt2))
        setPadding(dp(10), dp(10), dp(10), dp(10)); setBackgroundResource(R.drawable.bg_icon_btn); contentDescription = "닫기"
        setOnClickListener { dlg.dismiss() }
    }
    header.addView(close, LinearLayout.LayoutParams(dp(40), dp(40)))
    root.addView(header, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
    dlg.setContentView(root, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
    dlg.window?.let { w ->
        w.setBackgroundDrawableResource(R.drawable.bg_sheet)
        w.setGravity(Gravity.BOTTOM)
        w.setLayout(minOf(widthPx, dp(560)), heightPx)
        w.setDimAmount(0.55f)
        w.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE or WindowManager.LayoutParams.SOFT_INPUT_STATE_HIDDEN)
    }
    return SheetParts(dlg, root, subView, header)
}

/** 목록이 비었을 때 가운데에 보여 주는 아이콘 + 글 */
internal fun Activity.emptyState(iconRes: Int, text: String): View {
    val c = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER }
    c.addView(ImageView(this).apply { setImageResource(iconRes); imageTintList = android.content.res.ColorStateList.valueOf(col(R.color.txt3)) }, LinearLayout.LayoutParams(dp(44), dp(44)))
    c.addView(TextView(this).apply { this.text = text; textSize = 14f; setTextColor(col(R.color.txt3)) },
        LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(10) })
    return c
}

/** 알약 모양 버튼(배경 리소스는 bg_btn_*) */
internal fun Activity.pillButton(text: String, bgRes: Int, textColor: Int, heightDp: Int = 50, sizeSp: Float = 16f): TextView = TextView(this).apply {
    this.text = text; textSize = sizeSp; setTypeface(null, Typeface.BOLD); setTextColor(textColor); gravity = Gravity.CENTER
    setBackgroundResource(bgRes); isClickable = true; isFocusable = true; minHeight = dp(heightDp); setPadding(dp(18), 0, dp(18), 0)
}
