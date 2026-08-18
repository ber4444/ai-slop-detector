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

val USAGE = "Usage: kotlin slop-detector.main.kts <archive.webarchive> [--images] [--verbose]"
val KNOWN_FLAGS = setOf("--images", "--verbose")

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

fun percent(value: Double): String = String.format(Locale.ROOT, "%.1f%%", value * 100)

fun JsonObject.text(key: String): String? = this[key]?.jsonPrimitive?.contentOrNull
fun JsonObject.number(key: String): Double? = this[key]?.jsonPrimitive?.doubleOrNull
fun JsonObject.whole(key: String): Int? = this[key]?.jsonPrimitive?.intOrNull
fun JsonObject.strings(key: String): List<String> =
    this[key]?.jsonArray?.mapNotNull { it.jsonPrimitive.contentOrNull } ?: emptyList()

fun renderReport(payload: String, archive: String) {
    val report = try {
        Json.parseToJsonElement(payload).jsonObject
    } catch (error: Exception) {
        fail("The worker did not return a readable report: ${error.message}", 1)
    }

    val lines = mutableListOf("Archive: $archive", "", "Text detectors")
    val detectors = report["text"]?.jsonArray.orEmpty()
    if (detectors.isEmpty()) {
        lines += "  (no text detector results)"
    }
    for (element in detectors) {
        val detector = element.jsonObject
        val name = detector.text("name") ?: "unknown"
        val score = detector.number("score")
        val label = detector.text("label") ?: ""
        val detail = detector.text("detail")
        if (score == null) {
            // A detector that could not run explains itself over as many lines
            // as its error needs, rather than being squeezed into parentheses.
            lines += "  $name: unavailable"
            detail?.lines()?.filter { it.isNotBlank() }?.forEach { lines += "      ${it.trim()}" }
        } else {
            val head = "  $name: ${percent(score)} — $label"
            lines += if (detail.isNullOrBlank()) head else "$head ($detail)"
        }
    }

    val images = report["images"]?.jsonArray.orEmpty().map { it.jsonObject }
    if (images.isNotEmpty()) {
        lines += ""
        lines += "Image detectors"
        for (image in images) {
            val index = image.whole("index") ?: 0
            val skipped = image.text("skipped_reason")
            val score = image.number("score")
            lines += when {
                skipped != null -> "  Image $index: skipped ($skipped)"
                score != null -> "  Image $index: ${percent(score)} — ${image.text("label") ?: ""}"
                else -> "  Image $index: unavailable"
            }
        }

        val flagged = images.filter { it.strings("metadata_flags").isNotEmpty() }
        if (flagged.isNotEmpty()) {
            lines += ""
            lines += "Metadata heuristics"
            for (image in flagged) {
                for (flag in image.strings("metadata_flags")) {
                    lines += "  Image ${image.whole("index") ?: 0}: $flag"
                }
            }
        }
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
if (positional.size != 1) fail(USAGE, 2)
val archive = positional.single()
if (!archive.endsWith(".webarchive")) fail("Expected a .webarchive file: $archive", 2)

if (!skipBootstrap) bootstrapPython()

val python = System.getenv("SLOP_DETECTOR_PYTHON") ?: venvPython.path
val workerDirectory = File(repoRoot, "worker")
val command = listOf(python, "-m", "slop_detector.main", "--archive", File(archive).absolutePath) +
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

renderReport(payload, archive)
