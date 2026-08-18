#!/usr/bin/env kotlin
@file:DependsOn("org.jetbrains.kotlinx:kotlinx-serialization-json-jvm:1.8.1")

import java.io.File
import java.util.Locale
import kotlin.system.exitProcess
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

val USAGE = """
Usage: kotlin slop-detector.main.kts <file|glob|directory>... [--verbose]

  Accepts Markdown (.md, .markdown, .mdown, .mkd) files.
  Give several files, a glob such as '*.md', or a directory to survey a corpus:
  each file is scored on its own, and the report shows how many files each
  detector flags, with a histogram. --verbose lists every file's score.
""".trim()
val KNOWN_FLAGS = setOf("--verbose")

val repoRoot: File = File(System.getProperty("user.dir")).absoluteFile
val skipBootstrap: Boolean = System.getenv("SLOP_DETECTOR_SKIP_BOOTSTRAP") == "1"
val venvPython: File = File(repoRoot, ".venv/bin/python3")
val modelCache: File = File(repoRoot, ".model-cache")

fun fail(message: String, code: Int): Nothing {
    System.err.println(message)
    exitProcess(code)
}

fun run(command: List<String>, directory: File, description: String) {
    val exit = ProcessBuilder(command)
        .directory(directory)
        .redirectOutput(ProcessBuilder.Redirect.INHERIT)
        .redirectError(ProcessBuilder.Redirect.INHERIT)
        .start()
        .waitFor()
    if (exit != 0) fail("$description failed with exit code $exit", 1)
}

/** Create the project-local virtual environment and install worker requirements. */
fun bootstrapPython() {
    val requirements = File(repoRoot, "worker/requirements.txt")
    if (!requirements.isFile) fail("Missing ${requirements.path}", 1)

    if (!venvPython.isFile) {
        System.err.println("Creating the project-local Python environment in .venv ...")
        run(listOf("python3", "-m", "venv", ".venv"), repoRoot, "python3 -m venv .venv")
    }

    val stamp = File(repoRoot, ".venv/.requirements-stamp")
    val expected = "${requirements.length()}:${requirements.lastModified()}"
    if (!stamp.isFile || stamp.readText().trim() != expected) {
        System.err.println("Installing worker requirements (first run downloads packages) ...")
        run(
            listOf(venvPython.path, "-m", "pip", "install", "--quiet", "--upgrade", "pip"),
            repoRoot,
            "pip upgrade",
        )
        run(
            listOf(venvPython.path, "-m", "pip", "install", "--quiet", "-r", requirements.path),
            repoRoot,
            "pip install -r worker/requirements.txt",
        )
        stamp.writeText(expected)
    }
}

val TEXT_SUFFIXES = listOf(".md", ".markdown", ".mdown", ".mkd")

fun File.isSupported(): Boolean = TEXT_SUFFIXES.any { name.lowercase().endsWith(it) }

/** Expand one argument into the files it names: a glob, a directory, or a file. */
fun expand(argument: String): List<File> {
    val direct = File(argument)
    if (direct.isDirectory) {
        return direct.walkTopDown().filter { it.isFile && it.isSupported() }.sortedBy { it.path }.toList()
    }
    if (!argument.contains('*') && !argument.contains('?')) return listOf(direct)

    // The shell leaves a quoted glob unexpanded, so expand it here rather than
    // making the user remember which form they used.
    val pattern = File(argument)
    val parent = pattern.parentFile ?: File(".")
    val matcher = java.nio.file.FileSystems.getDefault().getPathMatcher("glob:${pattern.name}")
    return (parent.listFiles() ?: emptyArray())
        .filter { it.isFile && matcher.matches(File(it.name).toPath()) }
        .sortedBy { it.path }
}

fun percent(value: Double): String = String.format(Locale.ROOT, "%.1f%%", value * 100)

fun JsonObject.text(key: String): String? = this[key]?.jsonPrimitive?.contentOrNull
fun JsonObject.number(key: String): Double? = this[key]?.jsonPrimitive?.doubleOrNull
fun JsonObject.whole(key: String): Int? = this[key]?.jsonPrimitive?.intOrNull
fun JsonObject.strings(key: String): List<String> =
    this[key]?.jsonArray?.mapNotNull { it.jsonPrimitive.contentOrNull } ?: emptyList()

val EDITLENS = "EditLens"

class FileScore(val name: String, val score: Double?)

fun JsonObject.fileScores(): List<FileScore> =
    this["file_scores"]?.jsonArray.orEmpty().map {
        val file = it.jsonObject
        FileScore(file.text("name") ?: "?", file.number("score"))
    }

/** A ten-bucket bar chart of per-file scores, 0-10% .. 90-100%. */
fun histogram(files: List<FileScore>): List<String> {
    val scored = files.mapNotNull { it.score }
    if (scored.isEmpty()) return emptyList()
    val buckets = IntArray(10)
    for (score in scored) buckets[minOf(9, (score * 10).toInt())]++
    val widest = buckets.max().coerceAtLeast(1)
    return buckets.mapIndexed { index, count ->
        val filled = (count.toDouble() / widest * 24).toInt()
        val bar = "█".repeat(filled) + if (count > 0 && filled == 0) "▏" else ""
        val range = "${index * 10}–${index * 10 + 10}%"
        "    %-8s %-24s %d".format(range, bar, count)
    }
}

fun renderUnavailable(name: String, detail: String?, lines: MutableList<String>) {
    // A detector that could not run explains itself over as many lines as its
    // error needs, rather than being squeezed into parentheses.
    lines += "$name: could not run"
    detail?.lines()?.filter { it.isNotBlank() }?.forEach { lines += "    ${it.trim()}" }
}

/** One file: a plain per-detector verdict, no distribution. */
fun renderSingle(detectors: List<JsonObject>, verbose: Boolean, lines: MutableList<String>) {
    lines += "What the text detectors think"
    for (detector in detectors) {
        val name = detector.text("name") ?: "unknown"
        val score = detector.number("score")
        val label = detector.text("label") ?: ""
        val detail = detector.text("detail")
        if (score == null) {
            renderUnavailable("  $name", detail, lines)
        } else if (verbose) {
            val head = "  $name: ${percent(score)} — $label"
            lines += if (detail.isNullOrBlank()) head else "$head ($detail)"
        } else {
            lines += "  $name: $label"
        }
    }
}

/** Many files: how many each detector flags, a histogram, and the flagged files. */
fun renderSurvey(detectors: List<JsonObject>, total: Int, verbose: Boolean, lines: MutableList<String>) {
    for (detector in detectors) {
        val name = detector.text("name") ?: "unknown"
        val score = detector.number("score")
        lines += ""
        if (score == null) {
            renderUnavailable(name, detector.text("detail"), lines)
            continue
        }
        val files = detector.fileScores()
        val scored = files.filter { it.score != null }

        if (name == EDITLENS) {
            lines += "$name — ${detector.text("label") ?: ""}"
        } else {
            val flagged = scored.filter { it.score!! >= 0.5 }
            lines += "$name flags ${flagged.size} of ${scored.size} files as likely AI-generated"
        }
        histogram(files).forEach { lines += it }

        if (name != EDITLENS) {
            val flagged = scored.filter { it.score!! >= 0.5 }.sortedByDescending { it.score }
            if (flagged.isNotEmpty()) {
                lines += "  files it flags:"
                flagged.forEach { lines += "    ${percent(it.score!!)}  ${File(it.name).name}" }
            }
        }
        if (verbose) {
            lines += "  every file, highest first:"
            scored.sortedByDescending { it.score }
                .forEach { lines += "    ${percent(it.score!!)}  ${File(it.name).name}" }
        }
        lines += "  (aggregate over all text pooled: ${percent(score)} — footnote only)"
    }
}

fun renderReport(payload: String, inputs: List<File>, verbose: Boolean) {
    val report = try {
        Json.parseToJsonElement(payload).jsonObject
    } catch (error: Exception) {
        fail("The worker did not return a readable report: ${error.message}", 1)
    }

    val detectors = report["text"]?.jsonArray.orEmpty().map { it.jsonObject }
    val lines = mutableListOf<String>()
    if (inputs.size == 1) {
        lines += "Read: ${inputs.single().path}"
        lines += ""
        renderSingle(detectors, verbose, lines)
    } else {
        lines += "Surveyed ${inputs.size} files, each scored on its own"
        renderSurvey(detectors, inputs.size, verbose, lines)
    }

    val notes = report["warnings"]?.jsonArray.orEmpty().mapNotNull { it.jsonPrimitive.contentOrNull }
    if (notes.isNotEmpty()) {
        lines += ""
        lines += "Notes:"
        notes.forEach { lines += "  - $it" }
    }

    println(lines.joinToString(System.lineSeparator()))
}

val options = args.toList()
if (options.contains("--help") || options.contains("-h")) {
    println(USAGE)
    exitProcess(0)
}

val unknown = options.filter { it.startsWith("-") && it !in KNOWN_FLAGS }
if (unknown.isNotEmpty()) fail("Unknown option: ${unknown.first()}\n$USAGE", 2)

val positional = options.filterNot { it.startsWith("-") }
if (positional.isEmpty()) fail(USAGE, 2)

val inputs = positional.flatMap { expand(it) }.distinctBy { it.absolutePath }
if (inputs.isEmpty()) fail("No matching files: ${positional.joinToString(" ")}", 2)
val unsupported = inputs.filterNot { it.isSupported() }
if (unsupported.isNotEmpty()) {
    fail(
        "Unsupported file type: ${unsupported.first().path}\n" +
            "Expected one of: ${TEXT_SUFFIXES.joinToString(", ")}",
        2,
    )
}

if (!skipBootstrap) bootstrapPython()

val python = System.getenv("SLOP_DETECTOR_PYTHON") ?: venvPython.path
val workerDirectory = File(repoRoot, "worker")
val command = listOf(python, "-m", "slop_detector.main") +
    inputs.flatMap { listOf("--input", it.absolutePath) } +
    options.filter { it in KNOWN_FLAGS }

val builder = ProcessBuilder(command)
    .directory(if (workerDirectory.isDirectory) workerDirectory else repoRoot)
    .redirectError(ProcessBuilder.Redirect.INHERIT)
builder.environment()["HF_HOME"] = modelCache.path

val process = try {
    builder.start()
} catch (error: Exception) {
    fail("Could not start the Python worker ($python): ${error.message}", 1)
}
val payload = process.inputStream.bufferedReader().readText()
val code = process.waitFor()
if (code != 0) exitProcess(code)

renderReport(payload, inputs, options.contains("--verbose"))
