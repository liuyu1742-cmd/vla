"""Produce the evidence-aligned, simplified midterm version of Chapter 5."""

from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Cm
from docx.text.paragraph import Paragraph


MIDTERM_OBJECTS = {
    "5.2.1": [],
    "5.2.2": [],
    "5.2.3": [],
    "5.2.4": [],
    "5.2.5": ["娴风坏", "娓呮磥鍒?, "娓呮磥鍠烽浘鐡?],
    "5.2.6": ["閲忔澂", "鎵撹泲鍣?, "鎿€闈㈡潠", "鑼跺６", "璋冨懗缃?],
    "5.2.7": [],
    "5.2.8": ["姘寸摱", "鐩掕楗枡", "缃愯鐗?, "椹厠鏉?],
    "5.2.9": ["瀵嗗皝淇濋矞鐩?, "鏀剁撼鐩?],
    "5.2.10": ["鍐扮鎶藉眽", "鍐扮闂?, "寰尝鐐夐棬", "鎶藉眽", "姗辨煖闂?, "娲楃鏈洪棬"],
    "5.2.11": ["铚＄儧"],
    "5.2.12": ["鍜栧暋鏉?, "澶瑰瓙", "鎶惃鍒€", "鏈ㄥ嫼", "姘寸綈", "姹ゅ嫼", "婊ょ泦", "閾濈當绾?],
    "5.2.13": [],
    "5.2.14": [],
    "5.2.15": ["鍥轰綋棣欑殏", "鐨傛恫鍣?],
}


TASK_TITLES = {
    "5.2.1": "瀹剁數缁煎悎绠＄悊浠诲姟",
    "5.2.2": "鐜璋冭妭浠诲姟",
    "5.2.3": "鏅烘収瀹夐槻浠诲姟锛堝叆鍙ｅ畨鍏級",
    "5.2.4": "鏅烘収瀹夐槻浠诲姟锛堢幆澧冨畨鍏級",
    "5.2.5": "娓呮磥鐢ㄥ搧鍙栨斁涓庡綊浣嶄换鍔?,
    "5.2.6": "鍘ㄦ埧鍣ㄥ叿鍗曚欢褰掍綅浠诲姟",
    "5.2.7": "鍏绘姢绠＄悊浠诲姟",
    "5.2.8": "鐗╁搧閫掗€佷换鍔?,
    "5.2.9": "鏁寸悊鏀剁撼浠诲姟",
    "5.2.10": "鍌ㄧ墿璁炬柦寮€鍚堜换鍔?,
    "5.2.11": "瀹ゅ唴鎽嗘斁浠诲姟",
    "5.2.12": "椁愬叿涓庡鍣ㄦ憜鏀句换鍔?,
    "5.2.13": "琛ｇ墿闉嬬被涓庣巹鍏冲綊浣嶄换鍔?,
    "5.2.14": "宸ヤ綔瀛︿範鍖烘湇鍔′换鍔?,
    "5.2.15": "鍗荡鐢ㄥ搧涓庝釜浜哄崼鐢熸湇鍔′换鍔?,
}


_DETAILS = {
    "5.2.1": "闈㈠悜鐓ф槑銆佺┖璋冦€佺數瑙嗕互鍙婂帹鍗數鍣ㄧ殑鑱旂綉鎺у埗锛岄噸鐐规槸鎶婄敤鎴风殑鑷劧璇█闇€姹傝浆鎹负鏄庣‘鐨勮澶囨寚浠わ紝骞惰鍙栨墽琛屽悗鐨勫紑鍏炽€佹ā寮忔垨娓╁害鐘舵€併€備腑鏈熼獙鏀朵笉瑕佹眰鏈烘鑷傛帴瑙︽寜閿紝鑰岄噰鐢ㄦ櫤鎱у灞呮帴鍙ｅ畬鎴愬懡浠や笅鍙戙€佺姸鎬佸洖璇诲拰鏃ュ織鐣欏瓨銆?,
    "5.2.2": "鍥寸粫瀹ゅ唴娓╁害銆佹箍搴︺€佺┖姘旇川閲忓拰鍏夌収绛夎垝閫傚害鍙傛暟寮€灞曠洃娴嬩笌鑱斿姩銆傜郴缁熷厛璇诲彇浼犳劅鍣ㄦ暟鍊硷紝鍐嶄緷鎹槇鍊笺€佸畾鏃惰鍒欐垨鐢ㄦ埛鍋忓ソ鍚戠┖璋冦€佸姞婀垮櫒銆佸噣鍖栧櫒鍜岀伅鍏峰彂閫佽皟鑺傛寚浠わ紝閬垮厤鎶婂鏉傜殑鐜寤烘ā浣滀负涓湡蹇呮祴鍐呭銆?,
    "5.2.3": "閽堝鍏ユ埛闂ㄣ€侀棬閿併€侀棬閾冨拰鎽勫儚澶寸瓑鍏ュ彛璁炬柦寮€灞曠姸鎬佹煡鐪嬨€佽瀹㈢‘璁や笌甯冮槻鎺у埗銆傞獙鏀舵椂浠ラ棬閿佺姸鎬佸洖璇汇€侀棬閾冧簨浠惰褰曞拰鍛婅鎺ㄩ€佷负涓伙紝涓嶈姹傛満鍣ㄤ汉鍦ㄧ湡瀹為棬鍙ｈ繘琛岄暱璺濈宸￠€绘垨澶勭悊闄岀敓浜轰氦浜掋€?,
    "5.2.4": "閽堝鐑熼浘銆佺噧姘斻€佹紡姘村拰绱ф€ユ寜閽瓑瀹ゅ唴椋庨櫓婧愯繘琛屼簨浠剁洃娴嬨€佸憡璀﹀垎绾у拰娑堟伅鑱斿姩銆傜郴缁熶互浼犳劅鍣ㄤ簨浠躲€佽澶囩姸鎬佷笌鍛婅鏃ュ織褰㈡垚闂幆锛涗腑鏈熷彧楠岃瘉姝ｅ父瑙﹀彂銆佺姸鎬佸洖璇诲拰閫氱煡娴佺▼锛屼笉鎶婂缃湡瀹炲嵄闄╀綔涓哄疄楠屾潯浠躲€?,
    "5.2.5": "灏嗗師鍏堟秹鍙婃摝鎷建杩广€佸湴闈㈣鐩栧拰姹℃笉鍒ゆ柇鐨勬竻娲佹湇鍔℃敹缂╀负娓呮磥鐢ㄥ搧鐨勫崟浠跺彇鏀句笌褰掍綅銆傞€夌敤娴风坏銆佹竻娲佸埛鍜屾竻娲佸柗闆剧摱绛夊凡鏈夐厤瀵规牱鏈紝鍙獙璇佷粠鏌滀綋鎴栧彴闈㈡姄鍙栧苟鏀捐嚦鍥哄畾鐩爣浣嶏紝涓嶆妸鎺ヨЕ寮忔摝鎷€佽剰姹¤瘑鍒拰杩炵画杞ㄨ抗鍒楀叆涓湡娴嬭瘯銆?,
    "5.2.6": "灏嗙児楗€佸垏閰嶃€佸姞鐑瓑澶氶樁娈垫搷浣滆皟鏁翠负鍘ㄦ埧鍣ㄥ叿鐨勫崟浠跺綊浣嶃€傞噺鏉€佹墦铔嬪櫒銆佹搥闈㈡潠銆佽尪澹跺拰璋冨懗缃愬潎浣跨敤宸叉湁瑙嗚銆佽瑷€鍜屽姩浣滈厤瀵规暟鎹紝鎵ц鏂瑰紡闄愬畾涓哄彴闈笌鎶藉眽鎴栨┍鏌滀箣闂寸殑涓€娆℃姄鍙栧拰涓€娆℃斁缃紝涓嶅紑灞曞垏鑿溿€佺炕鐐掋€佺伀鍔涜皟鑺傛垨澶氶鏉愬垎绫汇€?,
    "5.2.7": "淇濈暀鍘熺増鍏绘姢绠＄悊鐨勮仈缃戣兘鍔涳紝浣嗗皢涓湡鍐呭闄愬畾涓烘彁閱掋€佺姸鎬佹煡鐪嬪拰鎺ュ彛鑱斿姩銆傛櫤鑳借姳鐩嗐€佺┖姘斿噣鍖栧櫒銆佹壂鍦版満鍣ㄤ汉鍙婃姤璀﹀櫒鐨勭淮鎶や俊鎭€氳繃璁惧鎺ュ彛璇诲彇锛屼笉瀹夋帓娴囨按銆佹媶鍗告护缃戙€佸畨瑁呰€楁潗鎴栨鐗╃梾铏璇嗗埆绛夐渶瑕侀澶栨劅鐭ュ拰绮剧粏鎿嶄綔鐨勭幆鑺傘€?,
    "5.2.8": "浠ユ煖浣撳埌鍙伴潰鐨勬槑纭捣缁堢偣楠岃瘉閫掗€佽兘鍔涳紝閫夊彇姘寸摱銆佺洅瑁呴ギ鏂欍€佺綈瑁呯墿鍜岄┈鍏嬫澂鍥涚被鍗曚欢瀵硅薄銆傛瘡鏉＄ず鏁欏潎鏄庣‘鐩爣瀵硅薄涓庢斁缃綅缃紝娴嬭瘯鍙瘎浼颁竴娆″彇鐗┿€佺煭璺濈绉诲姩鍜屽浐瀹氱偣鏀剧疆锛涗功绫嶅簥涓婅嚦搴婂ご鏌滃強鍐扮鍒板鍘呯殑缁勫悎璺嚎淇濈暀涓哄悗缁墿灞曟牱鏈€?,
    "5.2.9": "灏嗘暣鐞嗘敹绾抽檺瀹氫负鍗曚欢瀹瑰櫒褰掍綅锛屼娇鐢ㄥ瘑灏佷繚椴滅洅鍜屾敹绾崇洅鐨勫彴闈㈣嚦姗辨煖鏍锋湰銆傝璁剧疆涓嶆秹鍙婂鐗╀綋鎺掑簭銆佹娊灞夊唴绌洪棿瑙勫垝鎴栬法鎴块棿鎼繍锛屾満鍣ㄤ汉鍙渶鏍规嵁璇█鎸囦护璇嗗埆涓€涓洰鏍囧鍣ㄥ苟瀹屾垚涓€娆″畨鍏ㄦ斁缃紝浠庤€屽舰鎴愭槗澶嶇幇銆佹槗鍒ゅ畾鐨勪腑鏈熼棴鐜€?,
    "5.2.10": "浠ュ啺绠遍棬銆佸啺绠辨娊灞夈€佸井娉㈢倝闂ㄣ€佹櫘閫氭娊灞夈€佹┍鏌滈棬鍜屾礂纰楁満闂ㄤ负瀵硅薄锛岄獙璇佸崟涓€寮€鍚堟垨鎺ㄥ叆鍔ㄤ綔銆傛瘡椤逛换鍔￠兘鍏锋湁娓呮櫚鐨勭粓姝㈢姸鎬侊紝鍙緷鎹棬缂濄€佹娊灞変綅缃垨琛岀▼缁撴潫鍒ゅ畾鎴愬姛锛涗笉涓庡彇鐗┿€佸垎绫汇€佽澶囩▼搴忛€夋嫨绛夊姩浣滅粍鍚堬紝闄嶄綆绛栫暐鏃跺簭鍜屾姄鍙栫簿搴﹁姹傘€?,
    "5.2.11": "灏嗗師鏈夌簿缁嗗畨瑁呬换鍔℃敹缂╀负瀹ゅ唴鍗曚欢鎽嗘斁锛屽綋鍓嶄腑鏈熸牱鏈€夌敤铚＄儧浠庢┍鏌滃彇鍑哄苟鏀惧埌鍙伴潰銆傝浠诲姟淇濈暀瑙嗚瀹氫綅銆佹姄鍙栧拰鏀剧疆鐨勫熀鏈摼璺紝浣嗕笉閲囩敤娴锋姤鎮寕銆佸閽夊鍑嗐€佺浉鏈哄畨瑁呬笁鑴氭灦绛夊濮挎€佺簿搴﹀拰鎺ヨЕ绾︽潫瑕佹眰杈冮珮鐨勭ず鏁欎綔涓洪獙鏀跺璞°€?,
    "5.2.12": "闈㈠悜椁愬叿鍜屽帹鎴垮鍣ㄧ殑鍗曚欢鎽嗘斁锛岄€夊彇鍜栧暋鏉€佸す瀛愩€佹姭钀ㄥ垁銆佹湪鍕恒€佹按缃愩€佹堡鍕恒€佹护鐩嗗拰閾濈當绾哥瓑宸查厤瀵瑰璞°€備腑鏈熶换鍔′粎瑕佹眰鏌滀綋涓庡彴闈箣闂寸殑涓€娆″彇鏀撅紝寮鸿皟澶圭埅寮€鍚堟椂鏈哄拰鐩爣鍖哄煙绋冲畾钀芥斁锛涗笉绾冲叆閿呭唴杞Щ銆佺洏鍐呴鏉愭憜鏀惧拰澶氬櫒鐨跨粍鍚堝竷缃€?,
    "5.2.13": "琛ｇ墿闉嬬被涓庣巹鍏冲綊浣嶄繚鐣欎负鍚庣画鑳藉姏绫诲埆銆傜幇鏈夐瀷绫汇€佹瘺宸惧拰娲楄。鏈虹ず鏁欏寘鍚浠堕瀷鎺掑垪銆佹瘺宸惧垎绡垨娲楄。鏈烘竻娲楃瓑缁勫悎瑕佹眰锛岄毦浠ヤ綔涓虹ǔ瀹氱殑涓湡鍗曞洖鍚堟祴璇曘€備腑鏈熼樁娈靛彧淇濈暀鍏舵暟鎹储寮曞拰鏍囩鏍搁獙锛屽緟鍗曚欢琛ｇ墿鍏ョ鎴栧崟闉嬩笂鏋舵牱鏈ˉ榻愬悗鍐嶇撼鍏ユ墽琛岃瘎娴嬨€?,
    "5.2.14": "宸ヤ綔瀛︿範鍖烘湇鍔″悓鏍蜂繚鐣欎负鍚庣画鎵╁睍绫诲埆銆傚綋鍓嶆暟鎹腑鐨勬闈㈡暣鐞嗗悓鏃舵秹鍙婄數鑴戙€佹樉绀哄櫒銆侀敭鐩樸€侀紶鏍囥€佹枃浠跺す銆佺瑪璁版湰鍜屽姙鍏绛夊涓璞★紝鍖呭惈鎺掑簭銆佸爢鍙犲拰鐩稿浣嶇疆绾︽潫銆備负閬垮厤鎶婂瀵硅薄瑙勫垝璇啓鎴愬崟涓€鎶€鑳斤紝涓湡鍙睍绀烘暟鎹潵婧愪笌鍙浆鎹㈢姸鎬侊紝涓嶅皢鍏朵綔涓洪€氳繃鐜囬獙鏀朵换鍔°€?,
    "5.2.15": "鍗荡鐢ㄥ搧鏈嶅姟鐨勪腑鏈熺増鏈噰鐢ㄥ浐浣撻鐨傚拰鐨傛恫鍣ㄤ袱涓崟浠舵牱鏈紝鍒嗗埆瀵瑰簲鍙伴潰鑷虫┍鏌滄垨姗辨煖鑷冲彴闈㈢殑鏄庣‘鍙栨斁銆傜墮鍒枫€佺墮鑶忋€佹礂娑ゅ墏鍜屽崼鐢熷肪鐩掔殑鍘熷绀烘暀鍖呭惈澶氫欢鐢ㄥ搧鍒嗙被涓庡崼娴寸┖闂村竷缃紝鍥犳淇濈暀鍦ㄦ暟鎹粨搴撲腑锛屼絾涓嶈繘鍏ヤ腑鏈熼棴鐜紝閬垮厤浜х敓澶氭楠ゆ垚鍔熷垽瀹氭涔夈€?,
}


def _make_section(code: str) -> str:
    title = TASK_TITLES[code]
    detail = _DETAILS[code]
    return (
        f"{title}鏄潰鍚戝搴湇鍔″満鏅殑涓€涓嫭绔嬫妧鑳界被鍒紝寮鸿皟鍦ㄦ槑纭换鍔¤竟鐣屽唴瀹屾垚鍙鐜般€佸彲瑙傚療鍜屽彲鍒ゅ畾鐨勬搷浣溿€?
        f"{detail}"
        "鍦ㄦ暟鎹粍缁囦笂锛屼换鍔℃寚浠ゃ€佺涓€瑙嗚鎴栧閮≧GB瑙嗛浠ュ強鏈哄櫒浜哄姩浣滄爣绛句繚鎸佸悓婧愰厤瀵癸紝渚夸簬鍚庣画杞崲涓虹粺涓€璁粌鏍煎紡銆?
        "鍦ㄩ獙鏀惰璁′笂锛屽厛鍥哄畾瀵硅薄銆佽捣鐐广€佺粓鐐瑰拰鎴愬姛鐘舵€侊紝鍐嶈褰曟墽琛屾棩蹇椾笌瑙嗛缁撴灉锛涜繖鏍锋棦鑳藉睍绀鸿瑙夎瑷€鍔ㄤ綔妯″瀷鐨勫熀鏈兘鍔涳紝涔熻兘閬垮厤鎶婂皻鏈ǔ瀹氱殑澶嶆潅閾捐矾鎻愬墠绾冲叆缁撹銆?
        "褰撳悗缁鍔犳洿涓板瘜鐨勭ず鏁欏拰瀹夊叏绾︽潫鍚庯紝鍙湪涓嶆敼鍙樿浠诲姟瀹氫箟鐨勫墠鎻愪笅閫愭鎵╁睍瀵硅薄绉嶇被銆佺幆澧冩壈鍔ㄥ拰杩炵画鍔ㄤ綔闀垮害銆?
        "鏈樁娈靛舰鎴愮殑瀵硅薄銆佸姩浣滃拰缁撴灉璁板綍鍙綔涓哄悗缁鐜板疄楠屼笌鐗堟湰姣旇緝鐨勭粺涓€渚濇嵁銆?
    )


SECTION_TEXT = {code: _make_section(code) for code in TASK_TITLES}


FIGURE_SOURCES = {
    "5.2.1": ("5.2.10", "寰尝鐐夐棬"),
    "5.2.2": ("5.2.11", "铚＄儧"),
    "5.2.3": ("5.2.10", "姗辨煖闂?),
    "5.2.4": ("5.2.5", "娓呮磥鍠烽浘鐡?),
    "5.2.5": ("5.2.5", "娴风坏"),
    "5.2.6": ("5.2.6", "閲忔澂"),
    "5.2.7": ("5.2.10", "鍐扮闂?),
    "5.2.8": ("5.2.8", "姘寸摱"),
    "5.2.9": ("5.2.9", "瀵嗗皝淇濋矞鐩?),
    "5.2.10": ("5.2.10", "鎶藉眽"),
    "5.2.11": ("5.2.11", "铚＄儧"),
    "5.2.12": ("5.2.12", "鍜栧暋鏉?),
    "5.2.13": ("5.2.13", "闉嬪瓙"),
    "5.2.14": ("5.2.14", "绗旇鏈數鑴?),
    "5.2.15": ("5.2.15", "鍥轰綋棣欑殏"),
}


def section_length(text: str) -> int:
    return len("".join(text.split()))


def _task_dir(dataset_root: Path, code: str) -> Path:
    needle = "chapter5_" + code.replace(".", "_") + "_"
    return next(path for path in dataset_root.iterdir() if path.name.startswith(needle))


def standardize_figure_image(image_path: Path) -> Path:
    """Rewrite an extracted frame as a baseline RGB PNG accepted by python-docx."""
    from PIL import Image

    output = image_path.with_suffix(".png")
    with Image.open(image_path) as image:
        image.convert("RGB").save(output, format="PNG")
    return output


def extract_figures(dataset_root: Path, figure_dir: Path) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output = {}
    for code, (source_code, obj) in FIGURE_SOURCES.items():
        obj_dir = _task_dir(dataset_root, source_code) / obj
        video = next(obj_dir.rglob("*.mp4"))
        image = figure_dir / f"{code.replace('.', '_')}_{obj}.jpg"
        subprocess.run(["ffmpeg", "-y", "-ss", "0", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(image)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        output[code] = standardize_figure_image(image)
    return output


def _insert_after(paragraph: Paragraph, text: str = "") -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    result = Paragraph(new_p, paragraph._parent)
    result.style = paragraph.style
    if text:
        result.add_run(text)
    return result


def _clear_between(doc: Document, heading: Paragraph) -> Paragraph:
    body = doc.element.body
    children = list(body.iterchildren())
    index = children.index(heading._p)
    to_remove = []
    for child in children[index + 1 :]:
        if child.tag.endswith("}p") and "5.2." in "".join(child.itertext()) or child.tag.endswith("}p") and "5.3" in "".join(child.itertext()):
            break
        if child.tag.endswith("}tbl"):
            break
        to_remove.append(child)
    if not to_remove:
        return _insert_after(heading)
    first = Paragraph(to_remove[0], heading._parent)
    for extra in to_remove[1:]:
        body.remove(extra)
    return first


def _heading(doc: Document, code: str) -> Paragraph:
    for paragraph in doc.paragraphs:
        if paragraph.text.strip().startswith(code):
            return paragraph
    raise ValueError(f"missing heading {code}")


def _replace_sections(doc: Document, figures: dict[str, Path]) -> None:
    for ordinal, code in enumerate(TASK_TITLES, start=1):
        heading = _heading(doc, code)
        heading.clear()
        heading.add_run(f"{code} {TASK_TITLES[code]}")
        body = _clear_between(doc, heading)
        body.clear()
        body.add_run(SECTION_TEXT[code])
        figure_p = _insert_after(body)
        figure_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        figure_p.add_run().add_picture(str(figures[code]), width=Cm(11.5))
        source_code, obj = FIGURE_SOURCES[code]
        if code in {"5.2.1", "5.2.2", "5.2.3", "5.2.4", "5.2.7"}:
            caption_text = f"图5-{ordinal} {TASK_TITLES[code]}相关设备视觉上下文帧（对象：{obj}；网络控制以接口日志验收）"
        else:
            caption_text = f"图5-{ordinal} {TASK_TITLES[code]}相关VLA示教帧（对象：{obj}）"
        caption = _insert_after(figure_p, caption_text)
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _rewrite_summary(doc: Document) -> None:
    changes = {
        "5.2.5": ("娓呮磥鐢ㄥ搧鍙栨斁涓庡綊浣?, "娴风坏銆佹竻娲佸埛銆佹竻娲佸柗闆剧摱", "浠呭崟浠跺彇鏀撅紝涓嶅惈鎿︽嫮杞ㄨ抗", "鈽呪槄鈽嗏槅鈽?),
        "5.2.6": ("鍘ㄦ埧鍣ㄥ叿鍗曚欢褰掍綅", "閲忔澂銆佹墦铔嬪櫒銆佹搥闈㈡潠銆佽尪澹躲€佽皟鍛崇綈", "浠呭彴闈⑩€旀娊灞?姗辨煖褰掍綅", "鈽呪槄鈽嗏槅鈽?),
        "5.2.7": ("鍏绘姢绠＄悊", "鏅鸿兘鑺辩泦銆佺┖姘斿噣鍖栧櫒銆佹壂鍦版満鍣ㄤ汉銆佹姤璀﹀櫒", "鎺ュ彛鐘舵€佹煡鐪嬩笌缁存姢鎻愰啋", "鈽呪槄鈽嗏槅鈽?),
        "5.2.8": ("鐗╁搧閫掗€?, "姘寸摱銆佺洅瑁呴ギ鏂欍€佺綈瑁呯墿銆侀┈鍏嬫澂", "浠呮煖浣撯€斿彴闈㈢殑鍗曚欢閫掗€?, "鈽呪槄鈽嗏槅鈽?),
        "5.2.9": ("鏁寸悊鏀剁撼", "瀵嗗皝淇濋矞鐩掋€佹敹绾崇洅", "浠呭崟浠跺鍣ㄥ綊浣?, "鈽呪槄鈽嗏槅鈽?),
        "5.2.10": ("鍌ㄧ墿璁炬柦寮€鍚?, "鍐扮闂ㄣ€佹娊灞夈€佹┍鏌滈棬銆佸井娉㈢倝闂ㄣ€佹礂纰楁満闂?, "鍗曚竴寮€鍚堟垨鎺ㄥ叆鍔ㄤ綔", "鈽呪槄鈽嗏槅鈽?),
        "5.2.11": ("瀹ゅ唴鎽嗘斁", "铚＄儧", "姗辨煖鈥斿彴闈㈠崟浠舵憜鏀?, "鈽呪槄鈽嗏槅鈽?),
        "5.2.12": ("椁愬叿涓庡鍣ㄦ憜鏀?, "鍜栧暋鏉€佹湪鍕恒€佹堡鍕恒€佹护鐩嗙瓑", "鍗曚欢鏌滀綋鈥斿彴闈㈡憜鏀?, "鈽呪槄鈽嗏槅鈽?),
        "5.2.13": ("琛ｇ墿闉嬬被涓庣巹鍏冲綊浣?, "闉嬪瓙銆佹瘺宸俱€佹礂琛ｆ満", "鍚庣画鍌ㄥ锛屼笉绾冲叆涓湡闂幆", "鈽呪槄鈽呪槅鈽?),
        "5.2.14": ("宸ヤ綔瀛︿範鍖烘湇鍔?, "鏂囦欢澶广€佺瑪璁版湰鐢佃剳銆侀敭鐩樸€侀紶鏍囩瓑", "鍚庣画鍌ㄥ锛屼笉绾冲叆涓湡闂幆", "鈽呪槄鈽呪槅鈽?),
        "5.2.15": ("鍗荡鐢ㄥ搧涓庝釜浜哄崼鐢熸湇鍔?, "鍥轰綋棣欑殏銆佺殏娑插櫒", "浠呭崟浠舵煖浣撯€斿彴闈㈠彇鏀?, "鈽呪槄鈽嗏槅鈽?),
    }
    for row in doc.tables[0].rows[1:]:
        code = row.cells[0].text.strip()
        if code in changes:
            title, objects, boundary, difficulty = changes[code]
            row.cells[1].text = title
            row.cells[3].text = objects
            row.cells[4].text = boundary
            row.cells[5].text = difficulty


def _caption_like(doc: Document, text: str) -> Paragraph:
    paragraph = doc.add_paragraph(text)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return paragraph


def _insert_paragraph_before_element(doc: Document, element, text: str) -> None:
    paragraph = _caption_like(doc, text)
    element.addprevious(paragraph._p)


def _insert_original_network_tables(doc: Document, original_doc: Document) -> None:
    reference = doc.tables[1]
    copied = [("表5-2 家电控制类技能（40种）", 18), ("表5-3 环境调节类技能（20种）", 19), ("表5-4 安防监控类技能（15种）", 20), ("表5-7 养护管理类技能（10种）", 23)]
    for caption, index in copied:
        table = copy.deepcopy(original_doc.tables[index]._tbl)
        reference._tbl.addprevious(table)
        _insert_paragraph_before_element(doc, table, caption)
    _insert_paragraph_before_element(doc, reference._tbl, "表5-8 已核验VLA物体操作样本（88种）")
def update_document(doc_path: Path, original_path: Path, dataset_root: Path, figure_dir: Path) -> None:
    doc = Document(doc_path)
    original = Document(original_path)
    figures = extract_figures(dataset_root, figure_dir)
    _rewrite_summary(doc)
    _replace_sections(doc, figures)
    _insert_original_network_tables(doc, original)
    doc.save(doc_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    update_document(args.doc, args.original, args.dataset_root, args.figure_dir)
    print("updated")


if __name__ == "__main__":
    main()

